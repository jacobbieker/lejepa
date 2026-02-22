import warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import timm, mlflow, hydra, tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
import glob
import polars
import pandas as pd
import zarr
import zarr.storage
import xarray as xr
import numpy as np
from einops import rearrange
from lejepa.geospatial import osgb_to_lon_lat, lon_lat_to_osgb


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=16,
            in_chans=11*3,
            # 16 variables, 12 time steps, all surface variables, add spatial embeddings of the lat/lon or OSB number, so would add another 4-16 channels
        )
        # Final prediction needs to be matching the solar prediction data, so 12 * 2 (30min resolution) or 12 * 12 (5min resolution)
        # 30 min data is more reliable, and matches more the accumulated generation
        # ReLU output as cant be negative generation values, and batchnorm to stabilize training
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        # TODO Should take the NWP input, and then metadata about the system to be predicted, then output the embeddings and the projections for the SIGReg loss
        # TODO First pass is just NWP input, and then we can add the metadata in a second pass if we have time
        # TODO other metadata includes time of day (although sorta in the NWP data),
        # TODO Could do a per-timestep embedding, then merge the timestep embeddings to predict generation
        # metadata is just the ID of the system, as that includes all the other metadata about the system, location, size, etc, and then can do a lookup embedding for that and add it to the NWP embedding, or concatenate it and pass through an MLP
        # TODO Problem can be done as categorical, by binning the generation values, and then doing a classification loss on top of the probe, which might be easier to train and more stable, and also matches more the real world use case of predicting generation ranges rather than exact values
        # Similar to MetNet, and train one timestep at a time, still ideally would have metadata about the site, but can start without it
        # Each example is a different forecast time and generation for that time. Can try both easily, single timestep or combined timesteps
        # Also can do it with a 3 timstep one, the one before, at, and after and predict the 3-5 timesteps around that point

        # TODO First pass will be 3 timesteps, and 1 output step, aligned in time with the center one
        N, V = x.shape[:2]
        emb = self.backbone(x)
        # TODO Add the metadata embedding here to emb

        # The projection is to a 2D image, single channel, so would want a 2D series output from projection for the regression, or categorical classification
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


def main():
    mlflow.start_run()
    batch_size = 256

    train_ds = SolarDataset("train")
    test_ds = SolarDataset("validation")
    train = DataLoader(
        train_ds, batch_size=batch_size, num_workers=8,
    )
    test = DataLoader(test_ds, batch_size=batch_size, num_workers=8)

    # modules and loss
    net = ViTEncoder(proj_dim=5280).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 5)).to(
        "cuda")  # TODO Change out features to categorical bins for prediction, from 0 to 1 generation, or single output features/multiple features one for each forecast timestep
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": 2e-3, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * 800
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled="cuda" == "cuda")
    # Training
    for epoch in range(1000):
        net.train(), probe.train()
        for i, (vs, y) in tqdm.tqdm(enumerate(train), total=50*len(train), desc=f"Epoch: {epoch} Train"):
            if i > 50*len(train):
                break
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs["metoffice"] # Ignore metadata for now
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(
                    0) - proj).square().mean()  # RMSE of the mean of the projection vs the projection -> for solar, this would need to be a series of points
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * 0.02 + inv_loss * (1 - 0.02)
                # TODO Change this for the regression/categorical loss, not sure what cfg.V is doing
                # cfg.V is the number of views for the model, in this case for ours it would be 1
                # TODO change the cross entropy loss to MSE loss for regression, or CRPS if doing noise injection and 2 models at a time?
                yhat = probe(emb.detach())
                probe_loss = F.mse_loss(yhat, y)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            mlflow.log_metrics(
                {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                }
            )

        # Evaluation
        net.eval(), probe.eval()
        errors = []
        with torch.inference_mode():
            for i, (vs, y) in enumerate(tqdm.tqdm(test, total=len(test), desc="Test")):
                if i > len(test):
                    break
                vs = vs["metoffice"] # Ignore metadata for now
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    error = (probe(net(vs)[0]) - y).square().mean().item()
                    errors.append(error)
        mlflow.log_metrics({"test/mse": np.mean(errors), "test/epoch": epoch})
        # Save to disk
        model_state_dict = net.state_dict()
        probe_state_dict = probe.state_dict()
        torch.save(model_state_dict, f"model_state_dict_{epoch=}.pth")
        torch.save(probe_state_dict, f"probe_state_dict_{epoch=}.pth")


class SolarDataset(torch.utils.data.IterableDataset):
    def __init__(self,  buffer_size: int = 32, split: str = "train"):
        zarr_files = sorted(list(glob.glob("/nvme/metoffice-solar/data/*/*/*/*.zarr.zip")))
        self.zarr_files = [f for f in zarr_files if "monthly" in f]
        self.ds = self.open_zarrs(self.zarr_files)
        # Calculate mean/stddev
        # Split if train vs validation
        self.means = xr.open_dataset("/home/jacob/Development/lejepa/scripts/metoffice_means.nc")
        self.stds = xr.open_dataset("/home/jacob/Development/lejepa/scripts/metoffice_stds.nc")
        if split == "train":
            self.train = True
            self.ds = self.ds.sel(time=slice(None, "2024-04-01"))
            # One off calc mean/stddev of the variables
            #stats = self.ds.sel(time=np.random.choice(self.ds.time.values, size=101, replace=False))
            #means = stats.mean(dim="time").load()
            #stds = stats.std(dim="time").load()
            #means.to_netcdf("metoffice_means.nc")
            #stds.to_netcdf("metoffice_stds.nc")
        else:
            self.train = False
            self.ds = self.ds.sel(time=slice("2024-04-01", "2025-04-01"))
        self.pv_files = sorted(list(glob.glob("/nvme/uk_pv/30_minutely/*/*/data.parquet")))
        self.df = polars.scan_parquet(self.pv_files, hive_partitioning=True)
        self.metadata = pd.read_csv("/nvme/uk_pv/metadata.csv")
        self.bad_data = pd.read_csv("/nvme/uk_pv/bad_data.csv")
        # Remove all bad data ones from metadata
        i1 = self.metadata.set_index("ss_id").index
        i2 = self.bad_data.set_index("ss_id").index
        self.metadata = self.metadata[~i1.isin(i2)]
        self.times = self.ds.time.values
        self.buffer_size = buffer_size
        self.split = split

    def __len__(self):
        return len(self.times) - 2

    def __iter__(self):
        # Random orders of things
        if self.train:
            time_index_ordering = np.random.choice(len(self.times) - 2, size=len(self.times) - 2, replace=False)
        else:
            time_index_ordering = list(range(1, len(self.times) - 1))
        for i in time_index_ordering:
            timestamp = self.times[i]
            examples = self.get_examples(pd.Timestamp(timestamp), num_examples=128)
            for example in examples:
                metoffice = example["metoffice"].drop_vars(["lambert_azimuthal_equal_area"], errors="ignore")
                # Normalize with the means/ std
                metoffice = (metoffice - self.means.mean()) / self.stds.mean()
                # Convert to tensor, and move channel to the end
                metoffice = torch.from_numpy(metoffice.to_array(dim="channel").transpose("time", "channel", "projection_y_coordinate", "projection_x_coordinate").values.astype(np.float32))
                metoffice = rearrange(metoffice, "T C X Y -> (T C) X Y")
                pv = example["pv"]
                metadata = example["metadata"]
                # NEed to change metadata to a consisten format: [lat, lon, orientation_sin, orientation_cos, tilt, kWp], and then convert to tensor
                metadata = [metadata["latitude_sin"].iloc[0],metadata["latitude_cos"].iloc[0], metadata["longitude_sin"].iloc[0], metadata["longitude_cos"].iloc[0], metadata["orientation_sin"].iloc[0], metadata["orientation_cos"].iloc[0], metadata["tilt"].iloc[0]]
                yield {"metoffice": metoffice, "metadata": torch.from_numpy(np.asarray(metadata))}, torch.from_numpy(pv).float()


    def open_zarrs(self, zarr_files: list[str]) -> xr.Dataset:
        dses = []
        for zarr_file in zarr_files:
            store = zarr.storage.ZipStore(zarr_file)
            ds = xr.open_zarr(store, consolidated=False, decode_timedelta=True)
            dses.append(ds)
        ds = xr.concat(dses, dim="time").isel(step=0)
        return ds

    def get_examples(self, time: pd.Timestamp, num_examples=100) -> list[dict[str, xr.Dataset | pd.DataFrame]]:
        # Add the hour before and after
        times = [time - pd.Timedelta(hours=1), time, time + pd.Timedelta(hours=1)]
        try:
            metoffice_data = self.ds.sel(time=times)
            start_time = metoffice_data.time.values[0]
            end_time = metoffice_data.time.values[-1]
            pv_data = self.df.filter((polars.col("datetime_GMT").dt.replace_time_zone(None) >= start_time) & (
                    polars.col("datetime_GMT").dt.replace_time_zone(None) <= end_time))
            ids = np.random.choice(self.metadata["ss_id"].unique(), size=num_examples, replace=False)
            pv_data = pv_data.filter(polars.col("ss_id").is_in(ids)).collect().to_pandas()
            metadata = self.metadata[self.metadata["ss_id"].isin(ids)]
            # For each of the IDs, get the corresponding metadata, and then return the metoffice data, the pv data, and the metadata for those IDs, as a list of dicts
            examples = []
            buffer_size = 16000
            for ss_id in ids:
                pv_system = pv_data[pv_data["ss_id"] == ss_id]
                system_metadata = metadata[metadata["ss_id"] == ss_id]
                osgb_x, osgb_y = lon_lat_to_osgb(system_metadata["longitude_rounded"].iloc[0], system_metadata["latitude_rounded"].iloc[0])
                metoffice_system_data = metoffice_data.sel(projection_x_coordinate=slice(osgb_x - buffer_size, osgb_x + buffer_size), projection_y_coordinate=slice(osgb_y - buffer_size, osgb_y + buffer_size))
                # Few sanity checks, of 5 time points, and 32x32 input
                # Check length of pv_system is 5
                if len(pv_system) != 5:
                    continue
                    # Check metoffice_system_data has the right shape, should be 3 time points, and then the spatial dimensions should be 32x32
                if metoffice_system_data.time.shape[0] != 3:
                    continue
                if metoffice_system_data.projection_x_coordinate.shape[0] < 16 or metoffice_system_data.projection_y_coordinate.shape[0] < 16:
                    continue
                pv_system_generation = pv_system["generation_Wh"].values / (750 * system_metadata["kWp"].iloc[0])
                # If over 1.5 for any of them, then skip as well
                if np.sum(pv_system_generation) > 5: # Over 5 is more than 1 per one generally
                    continue
                system_metadata["tilt"] = system_metadata["tilt"].fillna(0.0) / 90.0
                system_metadata["orientation"] = system_metadata["orientation"].fillna(0.0)
                # Need sin and cos of orientation since its circular
                system_metadata["orientation_sin"] = np.sin(np.radians(system_metadata["orientation"].iloc[0]))
                system_metadata["orientation_cos"] = np.cos(np.radians(system_metadata["orientation"].iloc[0]))
                # Do sin/cos of lat/lon again
                system_metadata["latitude_sin"] = np.sin(np.radians(system_metadata["latitude_rounded"].iloc[0]))
                system_metadata["latitude_cos"] = np.cos(np.radians(system_metadata["latitude_rounded"].iloc[0]))
                system_metadata["longitude_sin"] = np.sin(np.radians(system_metadata["longitude_rounded"].iloc[0]))
                system_metadata["longitude_cos"] = np.cos(np.radians(system_metadata["longitude_rounded"].iloc[0]))
                examples.append(
                    {
                        "metoffice": metoffice_system_data,
                        "pv": pv_system_generation,
                        "metadata": system_metadata,
                    }
                )

            return examples
        except Exception as e:
            print(f"Exception: {e}")
            return []


if __name__ == "__main__":
    main()