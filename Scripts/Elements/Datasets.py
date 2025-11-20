import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import deepcopy as cdc


class Dataset_unified(Dataset):
    def __init__(
        self,
        path_csvs,
        output_configs,
        select_years,
        select_geo,
        nested_variables,
        with_cuda=False,
        scaling_type="normalization",
        precomputed_scaling_params=None,
    ):
        """
        Base Dataset class. Reads configurations, data, computes scaling and prepares batch readings.

        Args:
            path_csvs: directory where the data can be found
            output_configs: dictionary with the specific configuration for the output (sectors or aggregate, emission accounting)
            select_years: years to include in the analysis
            select_geo: country codes to include in the analysis
            nested_variables: variables to include
            with_cuda: device
            scaling_type: type of data scaling to use
            precomputed_scaling_params: scaling parameters to use, if any
        """
        self.with_cuda = with_cuda

        # check if air emissions are to be studied in level or deltas
        if output_configs["mode"] == "level":
            emi_df = pd.read_csv(path_csvs + "air_emissions_yearly_full.csv")
        elif output_configs["mode"] == "difference":
            emi_df = pd.read_csv(path_csvs + "air_emissions_yearly_diff.csv")
        else:
            raise ValueError(f"Unsupported mode: {output_configs['mode']}")

        # filter countries
        emi_df = emi_df[emi_df["geo"].isin(select_geo)]

        # read chosen emission output setting
        if output_configs["output"] == "Sectors":
            emi_selection = [
                "HeatingCooling",
                "Industry",
                "Land",
                "Mobility",
                "Other",
                "Power",
            ]
        elif output_configs["output"] in ["Total", "TotalECON", "TotalHOUSE"]:
            emi_selection = output_configs["output"]
        else:
            raise ValueError(f"Unsupported output: {output_configs['output']}")

        # start with only keys
        result_df = emi_df[["geo", "year"]].copy()

        # read emission dataset
        if (
            output_configs["measure"] == "KG_HAB"
            or output_configs["measure"] == "THS_T"
        ):
            for sector in emi_selection:
                sector_columns = []
                for activity in output_configs["grouping_structure"][sector]:
                    pattern = f'air_emissions_yearly:{output_configs["emission_type"]}:{activity}:{output_configs["measure"]}'
                    sector_columns.extend(emi_df.filter(regex=pattern).columns)
                result_df[sector] = emi_df[sector_columns].sum(axis=1)
        elif output_configs["measure"] == "both":
            for measure in ["THS_T", "KG_HAB"]:
                for sector in emi_selection:
                    sector_columns = []
                    for activity in output_configs["grouping_structure"][sector]:
                        pattern = f'air_emissions_yearly:{output_configs["emission_type"]}:{activity}:{measure}'
                        sector_columns.extend(emi_df.filter(regex=pattern).columns)
                    result_df[f"{sector}_{measure}"] = emi_df[sector_columns].sum(
                        axis=1
                    )
        else:
            raise ValueError(f"Unsupported measure: {output_configs['measure']}")

        emi_df = result_df

        self.precomputed_scaling_params = (
            precomputed_scaling_params if precomputed_scaling_params else {}
        )
        use_computed = False
        if self.precomputed_scaling_params:
            use_computed = True

        # perform scaling
        for col in emi_df.columns:
            if col in ["geo", "year"]:
                continue
            if not use_computed:
                if scaling_type == "normalization":
                    emi_mean = emi_df[col].mean()
                    emi_std = emi_df[col].std()
                    self.precomputed_scaling_params[col] = {
                        "mean": emi_mean,
                        "std": emi_std,
                    }
                elif scaling_type == "maxmin":
                    emi_max = emi_df[col].max()
                    emi_min = emi_df[col].min()
                    self.precomputed_scaling_params[col] = {
                        "max": emi_max,
                        "min": emi_min,
                    }

            if scaling_type == "normalization":
                emi_df[col] = (
                    emi_df[col] - self.precomputed_scaling_params[col]["mean"]
                ) / self.precomputed_scaling_params[col]["std"]
            elif scaling_type == "maxmin":
                emi_df[col] = (
                    emi_df[col] - self.precomputed_scaling_params[col]["min"]
                ) / (
                    self.precomputed_scaling_params[col]["max"]
                    - self.precomputed_scaling_params[col]["min"]
                )

        # total variable blocks available
        input_variables = [
            "crops_livestock",
            "energy_consumption",
            "energy_taxes",
            "EV_data",
            "ghg_intensity_electricity_production",
            "heat_pumps",
            "land_use",
            "modal_split_transport",
            "Monthly_electricity_statistics",
            "Monthly_oil_price_statistics",
            "renewable_share",
            "solar_thermal_surface",
            "trade",
            "train_performance",
        ]
        context_variables = [
            "gdp_quarterly",
            "population",
            "Monthly_hdd_cdd",
            "climate",
        ]

        # List to hold the content of each dataset
        dfs = []
        # carbon prices are set separately, as they do not have a "geo" key
        carbon_prices_file = os.path.join(path_csvs, "carbon_prices_yearly.csv")
        carbon_prices_df = pd.read_csv(carbon_prices_file, decimal=",")
        carbon_prices_df = carbon_prices_df[carbon_prices_df["year"].isin(select_years)]
        carbon_prices_df = carbon_prices_df[
            [
                col
                for col in carbon_prices_df.columns
                if col in nested_variables or col == "year"
            ]
        ]

        # Loop through filtered_datasets and read each .csv file
        for dataset_name in input_variables:
            file_path = f"{path_csvs}{dataset_name}.csv"
            try:
                df = pd.read_csv(file_path)
                df = df[df["geo"].isin(select_geo)]
                df = df[
                    [
                        col
                        for col in df.columns
                        if col in nested_variables or col in ["geo", "time", "year"]
                    ]
                ]
                # handle special high frequency data separately
                if dataset_name in [
                    "Monthly_electricity_statistics",
                    "Monthly_oil_price_statistics",
                    "Monthly_hdd_cdd",
                    "gdp_quarterly",
                ]:
                    df["time"] = pd.to_datetime(df["time"])

                    df["year"] = df["time"].dt.year
                    df["month"] = df["time"].dt.month
                    df = df.drop(columns=["time"])
                    df = df.pivot(index=["geo", "year"], columns="month")
                    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
                    df.reset_index(inplace=True)
                df = df[df["year"].isin(select_years)]
                dfs.append(df)
            except FileNotFoundError:
                print(f"File {file_path} not found.")

        # merge dataset list
        input_df = dfs[0]
        for df in dfs[1:]:
            input_df = pd.merge(input_df, df, on=["geo", "year"], how="outer")
        input_df = pd.merge(input_df, carbon_prices_df, on=["year"], how="outer")
        input_nested_variables = [
            col for col in input_df.columns if col not in ["geo", "year"]
        ]

        # handle NA
        na_columns = input_df.columns[input_df.isna().all()].tolist()
        if na_columns:
            print(f"Removing fully NA columns: {na_columns}")
            input_df = input_df.drop(columns=na_columns)
            input_nested_variables = [
                var for var in input_nested_variables if var not in na_columns
            ]

        # perform normalization
        if scaling_type == "normalization":
            for col in input_nested_variables[
                :
            ]:  # Use a copy to safely modify the list
                if use_computed:
                    if col not in self.precomputed_scaling_params:
                        input_df = input_df.drop(columns=[col])
                        input_nested_variables.remove(col)
                        continue
                    mean = self.precomputed_scaling_params[col]["mean"]
                    std = self.precomputed_scaling_params[col]["std"]
                else:
                    mean = input_df[col].mean(skipna=True)
                    std = input_df[col].std(skipna=True)
                if std == 0 or pd.isna(std):
                    input_df = input_df.drop(columns=[col])
                    input_nested_variables.remove(col)
                else:
                    input_df[col] = (input_df[col] - mean) / std
                    self.precomputed_scaling_params[col] = {"mean": mean, "std": std}

        elif scaling_type == "maxmin":
            for col in input_nested_variables[
                :
            ]:  # Use a copy to safely modify the list
                if use_computed:
                    if col not in self.precomputed_scaling_params:
                        input_df = input_df.drop(columns=[col])
                        input_nested_variables.remove(col)
                        continue
                    min_val = self.precomputed_scaling_params[col]["min"]
                    max_val = self.precomputed_scaling_params[col]["max"]
                else:
                    min_val = input_df[col].min(skipna=True)
                    max_val = input_df[col].max(skipna=True)
                if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                    input_df = input_df.drop(columns=[col])
                    input_nested_variables.remove(col)
                else:
                    input_df[col] = (input_df[col] - min_val) / (max_val - min_val)
                    self.precomputed_scaling_params[col] = {
                        "min": min_val,
                        "max": max_val,
                    }

        # repeat for context dataset
        dfs = []
        for dataset_name in context_variables:
            file_path = f"{path_csvs}{dataset_name}.csv"
            try:
                df = pd.read_csv(file_path)
                df = df[df["geo"].isin(select_geo)]
                df = df[
                    [
                        col
                        for col in df.columns
                        if col in nested_variables or col in ["geo", "time", "year"]
                    ]
                ]
                # Extract key columns
                if dataset_name in [
                    "Monthly_electricity_statistics",
                    "Monthly_oil_price_statistics",
                    "Monthly_hdd_cdd",
                    "gdp_quarterly",
                ]:
                    df["time"] = pd.to_datetime(df["time"])
                    df["year"] = df["time"].dt.year
                    df["month"] = df["time"].dt.month
                    df = df.loc[df["month"].isin([1, 4, 7, 10])]
                    df = df.drop(columns=["time"])
                    df = df.pivot(index=["geo", "year"], columns="month")
                    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
                    df.reset_index(inplace=True)
                df = df[df["year"].isin(select_years)]
                dfs.append(df)
            except FileNotFoundError:
                print(f"File {file_path} not found.")

        context_df = dfs[0]
        for df in dfs[1:]:
            context_df = pd.merge(context_df, df, on=["geo", "year"], how="outer")
        context_nested_variables = [
            col for col in context_df.columns if col not in ["geo", "year"]
        ]

        na_columns = context_df.columns[context_df.isna().all()].tolist()
        if na_columns:
            print(f"Removing fully NA columns: {na_columns}")
            context_df = context_df.drop(columns=na_columns)
            context_nested_variables = [
                var for var in context_nested_variables if var not in na_columns
            ]

        if scaling_type == "normalization":
            for col in context_nested_variables[
                :
            ]:  # Use a copy to safely modify the list
                if use_computed:
                    if col not in self.precomputed_scaling_params:
                        context_df = context_df.drop(columns=[col])
                        context_nested_variables.remove(col)
                        continue
                    mean = self.precomputed_scaling_params[col]["mean"]
                    std = self.precomputed_scaling_params[col]["std"]
                else:
                    mean = context_df[col].mean(skipna=True)
                    std = context_df[col].std(skipna=True)
                if std == 0 or pd.isna(std):
                    context_df = context_df.drop(columns=[col])
                    context_nested_variables.remove(col)
                else:
                    context_df[col] = (context_df[col] - mean) / std
                    self.precomputed_scaling_params[col] = {"mean": mean, "std": std}

        elif scaling_type == "maxmin":
            for col in context_nested_variables[
                :
            ]:  # Use a copy to safely modify the list
                if use_computed:
                    if col not in self.precomputed_scaling_params:
                        context_df = context_df.drop(columns=[col])
                        context_nested_variables.remove(col)
                        continue
                    min_val = self.precomputed_scaling_params[col]["min"]
                    max_val = self.precomputed_scaling_params[col]["max"]
                else:
                    min_val = context_df[col].min(skipna=True)
                    max_val = context_df[col].max(skipna=True)
                if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                    context_df = context_df.drop(columns=[col])
                    context_nested_variables.remove(col)
                else:
                    context_df[col] = (context_df[col] - min_val) / (max_val - min_val)
                    self.precomputed_scaling_params[col] = {
                        "min": min_val,
                        "max": max_val,
                    }

        # now check consistency across datasets
        merged_df = pd.merge(input_df, context_df, on=["geo", "year"], how="inner")
        merged_df = pd.merge(emi_df, merged_df, on=["geo", "year"], how="inner")

        # Now we want to filter both datasets to keep only the rows that appear in the merged_df
        input_df_filtered = input_df[
            input_df.set_index(["geo", "year"]).index.isin(
                merged_df.set_index(["geo", "year"]).index
            )
        ]
        context_df_filtered = context_df[
            context_df.set_index(["geo", "year"]).index.isin(
                merged_df.set_index(["geo", "year"]).index
            )
        ]
        emi_df_filtered = emi_df[
            emi_df.set_index(["geo", "year"]).index.isin(
                merged_df.set_index(["geo", "year"]).index
            )
        ]

        # Ensure they have the same ordering as the merged dataset
        input_df_filtered = (
            input_df_filtered.set_index(["geo", "year"])
            .loc[merged_df.set_index(["geo", "year"]).index]
            .reset_index()
        )
        context_df_filtered = (
            context_df_filtered.set_index(["geo", "year"])
            .loc[merged_df.set_index(["geo", "year"]).index]
            .reset_index()
        )
        emi_df_filtered = (
            emi_df_filtered.set_index(["geo", "year"])
            .loc[merged_df.set_index(["geo", "year"]).index]
            .reset_index()
        )

        self.keys = merged_df[["geo", "year"]]
        self.input_df = input_df_filtered.drop(columns=["geo", "year"])
        self.context_df = context_df_filtered.drop(columns=["geo", "year"])
        self.emi_df = emi_df_filtered.drop(columns=["geo", "year"])

        self.input_variable_names = cdc(self.input_df.columns)
        self.context_variable_names = cdc(self.context_df.columns)
        self.emission_columns = cdc(self.emi_df.columns)

        self.input_df = torch.tensor(self.input_df.values, dtype=torch.float32)
        self.emi_df = torch.tensor(self.emi_df.values, dtype=torch.float32)
        self.context_df = torch.tensor(self.context_df.values, dtype=torch.float32)

        self.input_df = torch.nan_to_num(self.input_df)
        self.emi_df = torch.nan_to_num(self.emi_df)
        self.context_df = torch.nan_to_num(self.context_df)

        if with_cuda:
            self.input_df = self.input_df.to("cuda")
            self.emi_df = self.emi_df.to("cuda")
            self.context_df = self.context_df.to("cuda")

        self.index_map = {
            (self.keys.iloc[i, 0], self.keys.iloc[i, 1]): i
            for i in range(self.keys.shape[0])
        }

    def __len__(self):
        """

        Returns: total number of rows in all datasets combined

        """
        return self.input_df.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves geo and year keys from index, extracting current input and context, together with current emissions.
        Using index_map, extract index for the same country in the previous year, and if present extracts previous input and context. Else assumes no change from current year.
        Returns:
            -current input
            -current context
            -current emissions
            -previous input
            -previous context

        """
        geo = self.keys.iloc[idx, 0]
        year = self.keys.iloc[idx, 1]
        input_current = self.input_df[idx]
        context_current = self.context_df[idx]

        emissions = self.emi_df[idx]

        prev_idx = self.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = self.input_df[prev_idx]
            context_prev = self.context_df[prev_idx]
        else:
            input_prev = input_current
            context_prev = context_current
        return input_current, context_current, emissions, input_prev, context_prev


class Dataset_forecasting_latent(Dataset):
    def __init__(self, Full_dataset: Dataset_unified):
        """
        Dataset for training the latent forecasting model.
        Minor modifications, but mostly working on the Dataset_unified backbone
        Args:
            Full_dataset: Dataset_unified
        """
        self.full_dataset = Full_dataset

        self.keys = self.full_dataset.keys
        self.input_df = self.full_dataset.input_df
        self.context_df = self.full_dataset.context_df

        self.index_map = self.full_dataset.index_map

        # actual dataset length
        self.base_length = len(self.full_dataset)

    def __len__(self):
        """
        forecasting latent works with random samples, we atrificially inflate the size of the dataset by a factor of 5, not to have a unique random sample in each epoch.
        Has virtually the same impact of training for longer.
        """
        return self.base_length * 5

    def __getitem__(self, idx):
        """
        Retrieves geo and year keys from index, extracting current input and context, together with current emissions.
        Using index_map, extract index for the same country in the previous year and two years prior, and if present extracts previous input and context. Else assumes no change from current year.
        Returns:
            -current input
            -current context
            -previous input (t-1)
            -previous context (t-1)
            -past input (t-2)
            -past context (t-2)
        """
        # correct index for dataset inflation
        curidx = idx % self.base_length
        geo = self.keys.iloc[curidx, 0]
        year = self.keys.iloc[curidx, 1]

        input_current = self.input_df[curidx]
        context_current = self.context_df[curidx]

        prev_idx = self.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = self.input_df[prev_idx]
            context_prev = self.context_df[prev_idx]
        else:
            input_prev = input_current
            context_prev = context_current

        past_idx = self.index_map.get((geo, year - 2))
        if past_idx is not None:
            input_past = self.input_df[past_idx]
            context_past = self.context_df[past_idx]
        else:
            input_past = input_current
            context_past = context_current

        return (
            input_current,
            context_current,
            input_prev,
            context_prev,
            input_past,
            context_past,
        )


class Dataset_predict_AR(Dataset_unified):
    """
    Adapted class to predict emissions
    """

    def __getitem__(self, idx):
        """
        Needs to extract two contexts, but also two emissions. Emission prediction model outputs the increase in emissions, and as such needs also previous emissions, on top of current.
        """
        geo = self.keys.iloc[idx, 0]
        year = self.keys.iloc[idx, 1]
        input_current = self.input_df[idx]
        context_current = self.context_df[idx]

        emissions = self.emi_df[idx]

        prev_idx = self.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = self.input_df[prev_idx]
            context_prev = self.context_df[prev_idx]
            emissions_prev = self.emi_df[prev_idx]
        else:
            input_prev = input_current
            context_prev = context_current
            emissions_prev = emissions
        return (
            input_current,
            context_current,
            emissions,
            input_prev,
            context_prev,
            emissions_prev,
        )


class Dataset_Projections_2030(Dataset_unified):
    """
    Wrapper class for generating projections.
    """

    def __init__(self, Full_dataset: Dataset_unified):
        """
        Inherits from Dataset_unified. Update the context to include projections context variables.
        """
        self.__dict__ = cdc(Full_dataset.__dict__)
        self.update_context()

    def update_context(self, scaling_type="normalization"):
        context_variables = [
            "Data/full_timeseries/projections/gdp_quarterly.csv",
            "Data/full_timeseries/projections/population.csv",
            "Data/full_timeseries/projections/climate.csv",
        ]

        select_years = np.arange(2023, 2030 + 1)
        select_geo = [
            "AT",
            "BE",
            "BG",
            "HR",
            "CY",
            "CZ",
            "DK",
            "EE",
            "EL",
            "FI",
            "FR",
            "DE",
            "HU",
            "IE",
            "IT",
            "LV",
            "LT",
            "LU",
            "MT",
            "NL",
            "PL",
            "PT",
            "RO",
            "SK",
            "SI",
            "ES",
            "SE",
        ]

        dfs = []
        for file_path in context_variables:
            try:
                df = pd.read_csv(file_path)
                df = df[df["geo"].isin(select_geo)]
                # Extract key columns
                if file_path in [
                    "Data/full_timeseries/projections/gdp_quarterly.csv",
                    "Data/full_timeseries/projections/Monthly_hdd_cdd.csv",
                ]:
                    df["time"] = pd.to_datetime(df["time"])
                    df["year"] = df["time"].dt.year
                    df["month"] = df["time"].dt.month
                    df = df.loc[df["month"].isin([1, 4, 7, 10])]
                    df = df.drop(columns=["time"])
                    df = df.pivot(index=["geo", "year"], columns="month")
                    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
                    df.reset_index(inplace=True)
                df = df[
                    [
                        col
                        for col in df.columns
                        if col in self.context_variable_names
                        or col in ["geo", "time", "year"]
                    ]
                ]
                df = df[df["year"].isin(select_years)]
                dfs.append(df)
            except FileNotFoundError:
                print(f"File {file_path} not found.")

        context_df = dfs[0]
        for df in dfs[1:]:
            context_df = pd.merge(context_df, df, on=["geo", "year"], how="outer")
        if scaling_type == "normalization":
            for col in self.context_variable_names[
                :
            ]:  # Use a copy to safely modify the list
                if col not in self.precomputed_scaling_params:
                    context_df = context_df.drop(columns=[col])
                    self.context_variable_names.remove(col)
                    continue
                mean = self.precomputed_scaling_params[col]["mean"]
                std = self.precomputed_scaling_params[col]["std"]
                context_df[col] = (context_df[col] - mean) / std

        elif scaling_type == "maxmin":
            for col in self.context_variable_names[
                :
            ]:  # Use a copy to safely modify the list
                if col not in self.precomputed_scaling_params:
                    context_df = context_df.drop(columns=[col])
                    self.context_variable_names.remove(col)
                    continue
                min_val = self.precomputed_scaling_params[col]["min"]
                max_val = self.precomputed_scaling_params[col]["max"]
                context_df[col] = (context_df[col] - min_val) / (max_val - min_val)

        self.keys_proj = context_df[["geo", "year"]]
        self.context_df_proj = context_df.drop(columns=["geo", "year"])

        self.context_df_proj = torch.tensor(
            self.context_df_proj.values, dtype=torch.float32
        )

        self.context_df_proj = torch.nan_to_num(self.context_df_proj)

        if self.with_cuda:
            self.context_df_proj = self.context_df_proj.to("cuda")

        self.index_map_proj = {
            (self.keys_proj.iloc[i, 0], self.keys_proj.iloc[i, 1]): i
            for i in range(self.keys_proj.shape[0])
        }

    def __len__(self):
        raise ValueError(
            "This is just a wrapper, using lengths and such will be omega bugged."
        )

    def __getitem__(self, idx):
        raise ValueError(
            "This is just a wrapper, using __get_items__ and such will be omega bugged."
        )

    def getitem_Current_Next(self, idx):
        geo = self.keys_proj.iloc[idx, 0]
        year = self.keys_proj.iloc[idx, 1]
        context_current = self.context_df_proj[idx]
        next_idx = self.index_map_proj.get((geo, year + 1))
        context_next = self.context_df_proj[next_idx]
        return context_current, context_next

    def get_from_keys(self, geo, year):
        idx = self.index_map_proj.get((geo, year))
        return self.getitem_Current_Next(idx)

    def get_from_keys_shifted(self, geo, year):
        idx = self.index_map_proj.get((geo, year - 1))
        return self.getitem_Current_Next(idx)
