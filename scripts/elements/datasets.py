"""
Dataset classes for EU emission forecasting.

This module provides PyTorch Dataset implementations for loading and preprocessing
socioeconomic, energy, and emission data across EU member states. The datasets
support the three-stage training pipeline:

1. **DatasetUnified**: Base dataset for VAE training with paired current/previous
   year observations.

2. **DatasetForecasting**: Extended dataset for latent forecaster training with
   three consecutive years (t, t-1, t-2).

3. **DatasetPrediction**: Dataset for emission predictor training with emission
   labels and temporal pairs.

4. **DatasetProjections2030**: Wrapper for generating future projections using
   projected context variables.

Data Sources:
    - Eurostat: Emissions, energy consumption, economic indicators, transport
    - IEA: Electricity statistics, oil prices, EV data
    - FAO: Agriculture, land use
    - Copernicus ERA5: Climate variables
    - World Bank: Carbon prices

Example:
    >>> dataset = DatasetUnified(
    ...     path_csvs="Data/full_timeseries/",
    ...     output_configs=output_configs,
    ...     select_years=range(2010, 2024),
    ...     select_geo=["DE", "FR", "IT"],
    ...     nested_variables=variable_list,
    ... )
    >>> x_t, c_t, emissions, x_t1, c_t1 = dataset[0]
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetUnified(Dataset):
    """
    Base dataset for training VAE and downstream models.

    This dataset loads and preprocesses multiple data sources, performing:
    - Geographic and temporal filtering
    - Scaling (normalization or min-max)
    - Handling of monthly/quarterly data aggregation
    - Alignment across data sources
    - Temporal pairing (current + previous year)

    The dataset provides paired observations (x_t, x_{t-1}) which are essential
    for training models that learn temporal dynamics.

    Attributes:
        input_df: Tensor of input features, shape (n_samples, n_input_features).
        context_df: Tensor of context features, shape (n_samples, n_context_features).
        emi_df: Tensor of emission targets, shape (n_samples, n_sectors).
        keys: DataFrame with 'geo' and 'year' columns for each sample.
        index_map: Dict mapping (geo, year) tuples to dataset indices.
        input_variable_names: List of input feature names.
        context_variable_names: List of context feature names.
        emission_columns: List of emission sector names.
        precomputed_scaling_params: Dict of scaling parameters for each variable.

    Example:
        >>> dataset = DatasetUnified(
        ...     path_csvs="Data/full_timeseries/",
        ...     output_configs={"output": "Sectors", "measure": "KG_HAB", ...},
        ...     select_years=range(2010, 2024),
        ...     select_geo=["DE", "FR"],
        ...     nested_variables=["gdp_quarterly:MillionEUR", ...],
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> x_t, c_t, y, x_t1, c_t1 = dataset[0]
    """

    # Class-level constants for variable categorization
    INPUT_VARIABLE_SOURCES = [
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

    CONTEXT_VARIABLE_SOURCES = [
        "gdp_quarterly",
        "population",
        "Monthly_hdd_cdd",
        "climate",
    ]

    HIGH_FREQUENCY_SOURCES = [
        "Monthly_electricity_statistics",
        "Monthly_oil_price_statistics",
        "Monthly_hdd_cdd",
        "gdp_quarterly",
    ]

    def __init__(
        self,
        path_csvs: str,
        output_configs: dict,
        select_years: np.ndarray | range,
        select_geo: list[str],
        nested_variables: list[str],
        with_cuda: bool = False,
        scaling_type: str = "normalization",
        precomputed_scaling_params: dict | None = None,
    ):
        """
        Initializes the dataset by loading and preprocessing all data sources.

        Args:
            path_csvs: Directory path containing the CSV data files.
            output_configs: Configuration for emission outputs:
                - 'output': 'Sectors', 'Total', 'TotalECON', or 'TotalHOUSE'
                - 'emission_type': 'CO2' or 'GHG'
                - 'measure': 'KG_HAB', 'THS_T', or 'both'
                - 'mode': 'level' or 'difference'
                - 'grouping_structure': Dict mapping sectors to activities
            select_years: Years to include (e.g., range(2010, 2024)).
            select_geo: Country codes to include (e.g., ['DE', 'FR', 'IT']).
            nested_variables: List of variable names to include from data files.
            with_cuda: If True, move tensors to GPU.
            scaling_type: 'normalization' (z-score) or 'maxmin' (0-1 scaling).
            precomputed_scaling_params: Optional dict of pre-computed scaling
                parameters. If None, parameters are computed from this dataset.
                Use this to apply training set scaling to validation/test sets.

        Raises:
            ValueError: If output_configs contains unsupported values.
        """
        self.with_cuda = with_cuda
        self.scaling_type = scaling_type
        self.precomputed_scaling_params = precomputed_scaling_params or {}
        self._use_precomputed = bool(precomputed_scaling_params)

        # Load and process emission data
        emi_df = self._load_emissions(path_csvs, output_configs, select_geo)

        # Load and process input variables
        input_df, input_nested_variables = self._load_input_variables(
            path_csvs, select_geo, select_years, nested_variables
        )

        # Load and process context variables
        context_df, context_nested_variables = self._load_context_variables(
            path_csvs, select_geo, select_years, nested_variables
        )

        # Align all datasets and filter to common (geo, year) pairs
        self._align_datasets(emi_df, input_df, context_df)

        # Store variable names before converting to tensors
        self.input_variable_names = list(self.input_df.columns)
        self.context_variable_names = list(self.context_df.columns)
        self.emission_columns = list(self.emi_df.columns)

        # Convert to tensors
        self._convert_to_tensors()

        # Build index map for efficient temporal lookups
        self.index_map = {
            (self.keys.iloc[i, 0], self.keys.iloc[i, 1]): i
            for i in range(len(self.keys))
        }

    def _load_emissions(
        self, path_csvs: str, output_configs: dict, select_geo: list[str]
    ) -> pd.DataFrame:
        """
        Loads and processes emission data.

        Args:
            path_csvs: Path to data directory.
            output_configs: Emission configuration dictionary.
            select_geo: Countries to include.

        Returns:
            DataFrame with emission columns, indexed by geo and year.
        """
        # Select level or difference data
        if output_configs["mode"] == "level":
            emi_df = pd.read_csv(f"{path_csvs}air_emissions_yearly_full.csv")
        elif output_configs["mode"] == "difference":
            emi_df = pd.read_csv(f"{path_csvs}air_emissions_yearly_diff.csv")
        else:
            raise ValueError(f"Unsupported mode: {output_configs['mode']}")

        emi_df = emi_df[emi_df["geo"].isin(select_geo)]

        # Determine which sectors to aggregate
        if output_configs["output"] == "Sectors":
            sectors = [
                "HeatingCooling",
                "Industry",
                "Land",
                "Mobility",
                "Other",
                "Power",
            ]
        elif output_configs["output"] in ["Total", "TotalECON", "TotalHOUSE"]:
            sectors = [output_configs["output"]]
        else:
            raise ValueError(f"Unsupported output: {output_configs['output']}")

        # Aggregate emissions by sector
        result_df = emi_df[["geo", "year"]].copy()

        measures = (
            ["THS_T", "KG_HAB"]
            if output_configs["measure"] == "both"
            else [output_configs["measure"]]
        )

        for measure in measures:
            for sector in sectors:
                sector_columns = []
                for activity in output_configs["grouping_structure"][sector]:
                    pattern = f"air_emissions_yearly:{output_configs['emission_type']}:{activity}:{measure}"
                    sector_columns.extend(emi_df.filter(regex=pattern).columns)

                col_name = f"{sector}_{measure}" if len(measures) > 1 else sector
                result_df[col_name] = emi_df[sector_columns].sum(axis=1)

        # Apply scaling
        result_df = self._scale_dataframe(result_df, exclude_cols=["geo", "year"])

        return result_df

    def _load_input_variables(
        self,
        path_csvs: str,
        select_geo: list[str],
        select_years: np.ndarray | range,
        nested_variables: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Loads and processes input variable datasets.

        Args:
            path_csvs: Path to data directory.
            select_geo: Countries to include.
            select_years: Years to include.
            nested_variables: Variables to select.

        Returns:
            Tuple of (DataFrame, list of variable names).
        """
        dfs = []

        # Load carbon prices separately (no geo column)
        carbon_prices_df = pd.read_csv(
            f"{path_csvs}carbon_prices_yearly.csv", decimal=","
        )
        carbon_prices_df = carbon_prices_df[carbon_prices_df["year"].isin(select_years)]
        carbon_prices_df = carbon_prices_df[
            [
                c
                for c in carbon_prices_df.columns
                if c in nested_variables or c == "year"
            ]
        ]

        # Load each input variable source
        for source in self.INPUT_VARIABLE_SOURCES:
            file_path = f"{path_csvs}{source}.csv"
            df = self._load_single_source(
                file_path, source, select_geo, select_years, nested_variables
            )
            if df is not None:
                dfs.append(df)

        # Merge all input sources
        input_df = dfs[0]
        for df in dfs[1:]:
            input_df = pd.merge(input_df, df, on=["geo", "year"], how="outer")

        # Add carbon prices (broadcast to all geos)
        input_df = pd.merge(input_df, carbon_prices_df, on=["year"], how="outer")

        # Get variable names and handle NAs
        var_names = [c for c in input_df.columns if c not in ["geo", "year"]]
        input_df, var_names = self._handle_na_columns(input_df, var_names)

        # Apply scaling
        input_df = self._scale_dataframe(input_df, exclude_cols=["geo", "year"])

        return input_df, var_names

    def _load_context_variables(
        self,
        path_csvs: str,
        select_geo: list[str],
        select_years: np.ndarray | range,
        nested_variables: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Loads and processes context variable datasets.

        Args:
            path_csvs: Path to data directory.
            select_geo: Countries to include.
            select_years: Years to include.
            nested_variables: Variables to select.

        Returns:
            Tuple of (DataFrame, list of variable names).
        """
        dfs = []

        for source in self.CONTEXT_VARIABLE_SOURCES:
            file_path = f"{path_csvs}{source}.csv"
            df = self._load_single_source(
                file_path,
                source,
                select_geo,
                select_years,
                nested_variables,
                quarterly_aggregation=True,
            )
            if df is not None:
                dfs.append(df)

        # Merge all context sources
        context_df = dfs[0]
        for df in dfs[1:]:
            context_df = pd.merge(context_df, df, on=["geo", "year"], how="outer")

        # Get variable names and handle NAs
        var_names = [c for c in context_df.columns if c not in ["geo", "year"]]
        context_df, var_names = self._handle_na_columns(context_df, var_names)

        # Apply scaling
        context_df = self._scale_dataframe(context_df, exclude_cols=["geo", "year"])

        return context_df, var_names

    def _load_single_source(
        self,
        file_path: str,
        source_name: str,
        select_geo: list[str],
        select_years: np.ndarray | range,
        nested_variables: list[str],
        quarterly_aggregation: bool = False,
    ) -> pd.DataFrame | None:
        """
        Loads a single data source file.

        Args:
            file_path: Path to CSV file.
            source_name: Name of the data source.
            select_geo: Countries to include.
            select_years: Years to include.
            nested_variables: Variables to select.
            quarterly_aggregation: If True, aggregate quarterly data to selected months.

        Returns:
            Processed DataFrame or None if file not found.
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping.")
            return None

        df = df[df["geo"].isin(select_geo)]

        # Select only requested variables
        keep_cols = ["geo", "time", "year"]
        df = df[[c for c in df.columns if c in nested_variables or c in keep_cols]]

        # Handle high-frequency data
        if source_name in self.HIGH_FREQUENCY_SOURCES:
            df["time"] = pd.to_datetime(df["time"])
            df["year"] = df["time"].dt.year
            df["month"] = df["time"].dt.month

            if quarterly_aggregation:
                # Keep only quarterly observations (Jan, Apr, Jul, Oct)
                df = df[df["month"].isin([1, 4, 7, 10])]

            df = df.drop(columns=["time"])

            # Pivot to wide format with month suffix
            df = df.pivot(index=["geo", "year"], columns="month")
            df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
            df = df.reset_index()

        df = df[df["year"].isin(select_years)]
        return df

    def _handle_na_columns(
        self, df: pd.DataFrame, var_names: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Removes columns that are entirely NA.

        Args:
            df: Input DataFrame.
            var_names: List of variable names to check.

        Returns:
            Tuple of (cleaned DataFrame, updated variable names).
        """
        na_columns = df.columns[df.isna().all()].tolist()
        if na_columns:
            print(f"Removing fully NA columns: {na_columns}")
            df = df.drop(columns=na_columns)
            var_names = [v for v in var_names if v not in na_columns]
        return df, var_names

    def _scale_dataframe(
        self, df: pd.DataFrame, exclude_cols: list[str]
    ) -> pd.DataFrame:
        """
        Applies scaling to DataFrame columns.

        Args:
            df: Input DataFrame.
            exclude_cols: Columns to exclude from scaling.

        Returns:
            Scaled DataFrame.
        """
        use_precomputed = self._use_precomputed
        cols_to_scale = [c for c in df.columns if c not in exclude_cols]

        for col in cols_to_scale[:]:  # Copy to allow modification
            if use_precomputed:
                if col not in self.precomputed_scaling_params:
                    df = df.drop(columns=[col])
                    cols_to_scale.remove(col)
                    continue
                params = self.precomputed_scaling_params[col]
            else:
                params = self._compute_scaling_params(df[col])
                if params is None:
                    df = df.drop(columns=[col])
                    cols_to_scale.remove(col)
                    continue
                self.precomputed_scaling_params[col] = params

            df[col] = self._apply_scaling(df[col], params)

        return df

    def _compute_scaling_params(self, series: pd.Series) -> dict | None:
        """
        Computes scaling parameters for a single column.

        Args:
            series: Pandas Series to compute parameters for.

        Returns:
            Dict of scaling parameters, or None if column should be dropped.
        """
        if self.scaling_type == "normalization":
            mean = series.mean(skipna=True)
            std = series.std(skipna=True)
            if std == 0 or pd.isna(std):
                return None
            return {"mean": mean, "std": std}
        elif self.scaling_type == "maxmin":
            min_val = series.min(skipna=True)
            max_val = series.max(skipna=True)
            if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                return None
            return {"min": min_val, "max": max_val}
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

    def _apply_scaling(self, series: pd.Series, params: dict) -> pd.Series:
        """
        Applies scaling to a single column.

        Args:
            series: Pandas Series to scale.
            params: Dict of scaling parameters.

        Returns:
            Scaled Series.
        """
        if self.scaling_type == "normalization":
            return (series - params["mean"]) / params["std"]
        elif self.scaling_type == "maxmin":
            return (series - params["min"]) / (params["max"] - params["min"])
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling_type}")

    def _align_datasets(
        self,
        emi_df: pd.DataFrame,
        input_df: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> None:
        """
        Aligns datasets to common (geo, year) pairs.

        Args:
            emi_df: Emission DataFrame.
            input_df: Input features DataFrame.
            context_df: Context features DataFrame.

        Side effects:
            Sets self.keys, self.input_df, self.context_df, self.emi_df
        """
        # Find common (geo, year) pairs
        merged = pd.merge(input_df, context_df, on=["geo", "year"], how="inner")
        merged = pd.merge(emi_df, merged, on=["geo", "year"], how="inner")

        # Create index for filtering
        merged_index = merged.set_index(["geo", "year"]).index

        # Filter and align each dataset
        def filter_and_align(df: pd.DataFrame) -> pd.DataFrame:
            df_filtered = df[df.set_index(["geo", "year"]).index.isin(merged_index)]
            df_aligned = (
                df_filtered.set_index(["geo", "year"]).loc[merged_index].reset_index()
            )
            return df_aligned

        input_df = filter_and_align(input_df)
        context_df = filter_and_align(context_df)
        emi_df = filter_and_align(emi_df)

        # Store results
        self.keys = merged[["geo", "year"]]
        self.input_df = input_df.drop(columns=["geo", "year"])
        self.context_df = context_df.drop(columns=["geo", "year"])
        self.emi_df = emi_df.drop(columns=["geo", "year"])

    def _convert_to_tensors(self) -> None:
        """
        Converts DataFrames to PyTorch tensors and handles NaN values.

        Side effects:
            Converts self.input_df, self.context_df, self.emi_df to tensors.
        """
        self.input_df = torch.tensor(self.input_df.values, dtype=torch.float32)
        self.context_df = torch.tensor(self.context_df.values, dtype=torch.float32)
        self.emi_df = torch.tensor(self.emi_df.values, dtype=torch.float32)

        # Replace NaN with 0
        self.input_df = torch.nan_to_num(self.input_df)
        self.context_df = torch.nan_to_num(self.context_df)
        self.emi_df = torch.nan_to_num(self.emi_df)

        if self.with_cuda:
            self.input_df = self.input_df.to("cuda")
            self.context_df = self.context_df.to("cuda")
            self.emi_df = self.emi_df.to("cuda")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.input_df.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample with temporal pairing.

        For each sample at time t, also retrieves data from time t-1 (if available).
        If previous year data is not available, uses current year as fallback.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - input_current: Input features at t, shape (n_input_features,)
                - context_current: Context at t, shape (n_context_features,)
                - emissions: Emission targets at t, shape (n_sectors,)
                - input_prev: Input features at t-1, shape (n_input_features,)
                - context_prev: Context at t-1, shape (n_context_features,)
        """
        geo = self.keys.iloc[idx, 0]
        year = self.keys.iloc[idx, 1]

        input_current = self.input_df[idx]
        context_current = self.context_df[idx]
        emissions = self.emi_df[idx]

        # Look up previous year
        prev_idx = self.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = self.input_df[prev_idx]
            context_prev = self.context_df[prev_idx]
        else:
            # Fallback to current if no previous year available
            input_prev = input_current
            context_prev = context_current

        return input_current, context_current, emissions, input_prev, context_prev


class DatasetForecasting(Dataset):
    """
    Dataset for training the latent space forecaster.

    Extends DatasetUnified to provide three consecutive time steps (t, t-1, t-2)
    needed for the forecaster which predicts z_t from z_{t-1} and z_{t-2}.

    The dataset is artificially inflated 5x to provide more variation in
    random sampling across epochs.

    Attributes:
        full_dataset: Reference to the underlying DatasetUnified.
        keys: DataFrame with geo and year for each sample.
        input_df: Input feature tensor.
        context_df: Context feature tensor.
        index_map: Mapping from (geo, year) to index.
        base_length: Actual number of unique samples.

    Example:
        >>> base_dataset = DatasetUnified(...)
        >>> forecast_dataset = DatasetForecasting(base_dataset)
        >>> x_t, c_t, x_t1, c_t1, x_t2, c_t2 = forecast_dataset[0]
    """

    def __init__(self, full_dataset: DatasetUnified):
        """
        Initializes the forecasting dataset from a unified dataset.

        Args:
            full_dataset: Pre-built DatasetUnified instance.
        """
        self.full_dataset = full_dataset

        # Share references to avoid memory duplication
        self.keys = full_dataset.keys
        self.input_df = full_dataset.input_df
        self.context_df = full_dataset.context_df
        self.index_map = full_dataset.index_map
        self.base_length = len(full_dataset)

    def __len__(self) -> int:
        """
        Returns inflated dataset length (5x base).

        The inflation provides more random sampling variation per epoch
        without requiring additional training epochs.
        """
        return self.base_length * 5

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retrieves a sample with two previous time steps.

        Args:
            idx: Sample index (will be moduloed by base_length).

        Returns:
            Tuple of:
                - input_current: Input features at t
                - context_current: Context at t
                - input_prev: Input features at t-1
                - context_prev: Context at t-1
                - input_past: Input features at t-2
                - context_past: Context at t-2
        """
        # Map inflated index to actual index
        actual_idx = idx % self.base_length

        geo = self.keys.iloc[actual_idx, 0]
        year = self.keys.iloc[actual_idx, 1]

        input_current = self.input_df[actual_idx]
        context_current = self.context_df[actual_idx]

        # Get t-1
        prev_idx = self.index_map.get((geo, year - 1))
        if prev_idx is not None:
            input_prev = self.input_df[prev_idx]
            context_prev = self.context_df[prev_idx]
        else:
            input_prev = input_current
            context_prev = context_current

        # Get t-2
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


class DatasetPrediction(DatasetUnified):
    """
    Dataset for training the emission predictor.

    Extends DatasetUnified to also return previous year emissions, which
    are needed to compute emission deltas (the prediction target).

    The emission predictor predicts: Δemissions_t = emissions_t - emissions_{t-1}

    Example:
        >>> dataset = DatasetPrediction(...)
        >>> x_t, c_t, y_t, x_t1, c_t1, y_t1 = dataset[0]
        >>> delta = y_t - y_t1  # Training target
    """

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retrieves a sample with temporal pairing and emission history.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - input_current: Input features at t
                - context_current: Context at t
                - emissions: Emission targets at t
                - input_prev: Input features at t-1
                - context_prev: Context at t-1
                - emissions_prev: Emissions at t-1 (for computing delta)
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


class DatasetProjections2030(DatasetUnified):
    """
    Wrapper dataset for generating 2030 projections with projected context.

    This class extends a trained dataset with projected context variables
    (GDP, population, climate) for the years 2023-2030, enabling future
    emission projections.

    The projected context comes from external sources (e.g., OECD, UN
    population projections, climate models).

    Attributes:
        keys_proj: DataFrame with geo and year for projection samples.
        context_df_proj: Tensor of projected context features.
        index_map_proj: Mapping from (geo, year) to projection index.

    Example:
        >>> base_dataset = DatasetUnified(...)  # Train on historical data
        >>> proj_dataset = DatasetProjections2030(base_dataset)
        >>> c_t, c_next = proj_dataset.get_from_keys("DE", 2025)

    Note:
        Standard __len__ and __getitem__ raise errors to prevent misuse.
        Use get_from_keys() or getitem_Current_Next() instead.
    """

    PROJECTION_FILES = [
        "Data/full_timeseries/projections/gdp_quarterly.csv",
        "Data/full_timeseries/projections/population.csv",
        "Data/full_timeseries/projections/climate.csv",
    ]

    EU27_COUNTRIES = [
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

    def __init__(self, full_dataset: DatasetUnified):
        """
        Initializes projection dataset from a trained unified dataset.

        Args:
            full_dataset: Pre-trained DatasetUnified instance with scaling
                parameters that will be applied to projected data.
        """
        # Copy all attributes from the base dataset
        self.__dict__ = deepcopy(full_dataset.__dict__)
        self._load_projections()

    def _load_projections(self) -> None:
        """
        Loads and processes projected context variables for 2023-2030.

        Side effects:
            Sets self.keys_proj, self.context_df_proj, self.index_map_proj
        """
        select_years = np.arange(2023, 2031)
        dfs = []

        for file_path in self.PROJECTION_FILES:
            try:
                df = pd.read_csv(file_path)
                df = df[df["geo"].isin(self.EU27_COUNTRIES)]

                # Handle quarterly data
                if "gdp_quarterly" in file_path or "Monthly_hdd_cdd" in file_path:
                    df["time"] = pd.to_datetime(df["time"])
                    df["year"] = df["time"].dt.year
                    df["month"] = df["time"].dt.month
                    df = df[df["month"].isin([1, 4, 7, 10])]
                    df = df.drop(columns=["time"])
                    df = df.pivot(index=["geo", "year"], columns="month")
                    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
                    df = df.reset_index()

                # Keep only variables present in the training set
                keep_cols = [
                    c
                    for c in df.columns
                    if c in self.context_variable_names or c in ["geo", "year"]
                ]
                df = df[keep_cols]
                df = df[df["year"].isin(select_years)]
                dfs.append(df)

            except FileNotFoundError:
                print(f"Warning: Projection file {file_path} not found.")

        # Merge projection sources
        context_df = dfs[0]
        for df in dfs[1:]:
            context_df = pd.merge(context_df, df, on=["geo", "year"], how="outer")

        # Apply scaling using training set parameters
        for col in list(self.context_variable_names):
            if col not in self.precomputed_scaling_params:
                if col in context_df.columns:
                    context_df = context_df.drop(columns=[col])
                continue

            if col in context_df.columns:
                params = self.precomputed_scaling_params[col]
                context_df[col] = self._apply_scaling(context_df[col], params)

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
            for i in range(len(self.keys_proj))
        }

    def __len__(self) -> int:
        """Raises error - use specialized access methods instead."""
        raise ValueError(
            "DatasetProjections2030 is a wrapper class. "
            "Use get_from_keys() or getitem_Current_Next() instead."
        )

    def __getitem__(self, idx: int) -> Any:
        """Raises error - use specialized access methods instead."""
        raise ValueError(
            "DatasetProjections2030 is a wrapper class. "
            "Use get_from_keys() or getitem_Current_Next() instead."
        )

    def getitem_current_next(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets context for current and next year by projection index.

        Args:
            idx: Index into projection dataset.

        Returns:
            Tuple of (context_current, context_next).
        """
        geo = self.keys_proj.iloc[idx, 0]
        year = self.keys_proj.iloc[idx, 1]

        context_current = self.context_df_proj[idx]
        next_idx = self.index_map_proj.get((geo, year + 1))
        context_next = self.context_df_proj[next_idx]

        return context_current, context_next

    def get_from_keys(self, geo: str, year: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets context for current and next year by geographic code and year.

        Args:
            geo: Country code (e.g., 'DE', 'FR').
            year: Year for which to get context.

        Returns:
            Tuple of (context_current, context_next).
        """
        idx = self.index_map_proj.get((geo, year))
        return self.getitem_current_next(idx)

    def get_from_keys_shifted(
        self, geo: str, year: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets context shifted by one year (for forecaster input alignment).

        Args:
            geo: Country code.
            year: Target year (will look up year-1 as current).

        Returns:
            Tuple of (context_{year-1}, context_year).
        """
        idx = self.index_map_proj.get((geo, year - 1))
        return self.getitem_current_next(idx)
