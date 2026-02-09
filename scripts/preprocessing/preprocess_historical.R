# =============================================================================
# Historical Data Preprocessing Script
# =============================================================================
#
# This script preprocesses raw data from multiple sources (Eurostat, IEA, FAO,
# EEA, World Bank) into a standardized format for the EU emissions forecasting
# model.
# The operations of this script are reported, but not included in the pipeline,
# which directly downloads processed csv files stored in cloud.
#
# Input: Raw data files, assumed to be in data/raw/
# Output: Processed CSV files in data/full_timeseries/
#
# Data Sources:
#   - Eurostat: Emissions, energy, transport, population, GDP
#   - IEA: Electricity statistics, oil prices, EV data
#   - FAO: Agriculture, land use
#   - EEA: GHG intensity of electricity
#   - World Bank: Carbon prices
#   - Copernicus ERA5: Climate data
#
# Usage:
#   Rscript scripts/preprocessing/preprocess_historical.R
#
# Requirements:
#   install.packages(c("tidyverse", "readr", "readxl", "plyr", "reshape2",
#                      "countrycode", "data.table", "lubridate"))
#
# Author: Jacopo Ghirri
# License: MIT
# =============================================================================

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(readxl)
  library(plyr)
  library(reshape2)
  library(countrycode)
  library(data.table)
  library(lubridate)
})

# Set working directory to repository root (adjust if needed)
# setwd("path/to/eu-emissions-forecast")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# EU27 country codes (ISO 3166-1 alpha-2, with Greece as 'EL' per Eurostat)
EU27_COUNTRIES <- c(
  "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL",
  "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
  "SI", "ES", "SE"
)

# NACE sector codes and their readable names
NACE_CODES <- c(
  "A", "A02", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "TOTAL", "TOTAL_HH", "HH",
  "HH_HEAT", "HH_TRA", "HH_OTH"
)

NACE_NAMES <- c(
  "A-Agricolture,Forestry,Fishing", "A02-Forestry", "B-Mining",
  "C-Manufacturing", "D-Electricity,Gas,Steam,AirConditioning",
  "E-Water,Waste", "F-Construction", "G-Trade,VehicleRepair",
  "H-Transportation,Storage", "I-Accomodation,FoodService",
  "J-Information,Communication", "K-Finance,Insurance", "L-RealEstate",
  "M-Science,Technical,Professional", "N-Administrative",
  "O-PublicAdmin,Defence", "P-Education", "Q-Health",
  "R-Arts,Entertainement", "S-OtherService",
  "T-HouseholdasemployersActivities", "U-ExtraterritorialOrgsActivities",
  "Total", "AllNACEplusHousehold", "TotalHousehold",
  "HeatingCoolingbyHousehold", "TransportbyHousehold", "OtherbyHousehold"
)

# Input/output directories
RAW_DIR <- "data/raw"
OUTPUT_DIR <- "data/full_timeseries"

# Create output directory if it doesn't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

#' Reshape data from long to wide format with standardized column naming
#'
#' @param df Data frame in long format
#' @param id_vars Character vector of ID columns (typically c("geo", "year"))
#' @param selection_col Name of the column containing variable names
#' @param value_col Name of the column containing values
#' @return Data frame in wide format with geo and year as first two columns
reshape_to_wide <- function(df, id_vars = c("geo", "year"),
                            selection_col = "selection", value_col = "value") {
  # Create formula for acast

  formula_str <- paste(paste(id_vars, collapse = "+"), "~", selection_col)

  # Reshape to wide format
  wide_df <- as.data.frame(
    acast(as.formula(formula_str), value.var = value_col, data = df)
  )


  # Extract geo and year from row names (format: "GEO_YEAR")
  wide_df <- cbind(rownames(wide_df), rownames(wide_df), wide_df)
  colnames(wide_df)[1:2] <- id_vars
  wide_df[[id_vars[1]]] <- gsub("_.*", "", wide_df[[id_vars[1]]])
  wide_df[[id_vars[2]]] <- gsub(".*_", "", wide_df[[id_vars[2]]])
  rownames(wide_df) <- NULL

  return(wide_df)
}


#' Filter data to EU27 countries only
#'
#' @param df Data frame with a 'geo' column
#' @return Filtered data frame
filter_eu27 <- function(df) {
  df[df$geo %in% EU27_COUNTRIES, ]
}


#' Save processed data to CSV
#'
#' @param df Data frame to save
#' @param filename Output filename (without path)
#' @param output_dir Output directory path
save_processed <- function(df, filename, output_dir = OUTPUT_DIR) {
  filepath <- file.path(output_dir, filename)
  write.csv(df, filepath, row.names = FALSE)
  message(sprintf("Saved: %s (%d rows, %d columns)",
                  filepath, nrow(df), ncol(df)))
}


# =============================================================================
# Data Processing Functions
# =============================================================================

# -----------------------------------------------------------------------------
# Air Emissions (Eurostat)
# -----------------------------------------------------------------------------

process_air_emissions <- function() {
  message("\n--- Processing Air Emissions ---")

  df <- read_csv(
    file.path(RAW_DIR, "air_emissions_yearly.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  # Filter to EU27 and remove frequency column
  df <- df[df$geo %in% EU27_COUNTRIES, colnames(df) != "freq"]


  # Map NACE codes to readable names
  df$nace_r2 <- mapvalues(df$nace_r2, from = NACE_CODES, to = NACE_NAMES,
                          warn_missing = FALSE)

  # Rename columns
  colnames(df)[5:6] <- c("year", "value")

  # Create selection column: "air_emissions_yearly:POLLUTANT:SECTOR:UNIT"
  df$selection <- paste0(
    "air_emissions_yearly:",
    as.factor(df$airpol), ":",
    as.factor(df$nace_r2), ":",
    as.factor(df$unit)
  )

  # Reshape to wide format
  result <- reshape_to_wide(df)

  save_processed(result, "air_emissions_yearly_full.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Carbon Prices (World Bank)
# -----------------------------------------------------------------------------

process_carbon_prices <- function() {
  message("\n--- Processing Carbon Prices ---")

  df <- read_delim(
    file.path(RAW_DIR, "carbon_prices_yearly.csv"),
    delim = ";",
    escape_double = FALSE,
    col_types = cols(.default = col_character()),
    trim_ws = TRUE,
    show_col_types = FALSE
  )

  # Rename year column and clean column names
  colnames(df)[1] <- "year"
  colnames(df) <- gsub(" ", "_", colnames(df))

  # Convert to numeric (except year)
  for (col in colnames(df)[-1]) {
    df[[col]] <- as.numeric(gsub(",", ".", df[[col]]))
  }

  # Add prefix to column names for consistency
  colnames(df)[-1] <- paste0("carbon_price:", colnames(df)[-1])

  save_processed(df, "carbon_prices_yearly.csv")

  return(df)
}


# -----------------------------------------------------------------------------
# Energy Consumption (Eurostat)
# -----------------------------------------------------------------------------

process_energy_consumption <- function() {
  message("\n--- Processing Energy Consumption ---")

  df <- read_delim(
    file.path(RAW_DIR, "energy_consumption.tsv"),
    delim = "\t",
    col_types = cols(.default = col_double(), .default = col_character()),
    trim_ws = TRUE,
    show_col_types = FALSE
  )

  # Parse the first column which contains multiple fields
  colnames(df)[1] <- "combined"
  df$combined <- sub("A,TOTAL,", "", df$combined)

  # Extract geo, type, and unit from combined column
  df$geo <- gsub(".*,", "", df$combined)
  df$type <- gsub(",.*", "", df$combined)
  df$unit <- sub(",.*", "", sub(".*?,", "", df$combined))

  # Filter to EU27
  df$geo[df$geo == "EU27_2020"] <- "EU_27"
  df <- filter_eu27(df)

  # Create selection column
  df$selection <- paste0("energy_consumption:", df$type, ":", df$unit)

  # Reshape from wide to long, then back to wide with proper format
  year_cols <- grep("^[0-9]{4}$", colnames(df), value = TRUE)
  df_long <- melt(
    df[, c("selection", "geo", year_cols)],
    id.vars = c("selection", "geo"),
    variable.name = "year",
    value.name = "value"
  )

  result <- reshape_to_wide(df_long)

  save_processed(result, "energy_consumption.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Energy Taxes (Eurostat)
# -----------------------------------------------------------------------------

process_energy_taxes <- function() {
  message("\n--- Processing Energy Taxes ---")

  df <- read_csv(
    file.path(RAW_DIR, "energy_taxes.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  # Filter to EU27 and select relevant columns
  df <- filter_eu27(df)
  df <- df[, c("geo", "TIME_PERIOD", "OBS_VALUE")]
  colnames(df) <- c("geo", "year", "energy_taxes:MIOEUR")

  save_processed(df, "energy_taxes.csv")

  return(df)
}


# -----------------------------------------------------------------------------
# Electric Vehicle Data (IEA)
# -----------------------------------------------------------------------------

process_ev_data <- function() {
  message("\n--- Processing EV Data ---")

  df <- read_csv(
    file.path(RAW_DIR, "EV_data.csv"),
    show_col_types = FALSE
  )

  colnames(df)[1] <- "geo"

  # Standardize country names
  df$geo <- mapvalues(
    df$geo,
    from = c("Turkiye", "EU27"),
    to = c("Turkey", "EU_27"),
    warn_missing = FALSE
  )

  # Convert to ISO2 codes
  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c(
      "EU_27" = "EU_27", "Europe" = "Europe",
      "Other Europe" = "Other Europe",
      "Rest of the world" = "Rest of the world",
      "World" = "World"
    )
  )

  # Filter to historical data and EU27 countries
  df <- df[df$category == "Historical", ]
  df <- filter_eu27(df)

  # Create selection column: "EV_data:PARAMETER:MODE:POWERTRAIN:UNIT"
  df$selection <- paste0(
    "EV_data:",
    df$parameter, ":",
    df$mode, ":",
    df$powertrain, ":",
    df$unit
  )

  df <- df[, c("selection", "geo", "year", "value")]

  result <- reshape_to_wide(df)

  save_processed(result, "EV_data.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# GDP Quarterly (Eurostat)
# -----------------------------------------------------------------------------

process_gdp_quarterly <- function() {
  message("\n--- Processing GDP Quarterly ---")

  df <- read_csv(
    file.path(RAW_DIR, "gdp_quarterly.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  colnames(df) <- c("unit", "adjustment", "prod", "geo", "time", "value")

  # Convert country names to ISO2 codes
  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c("Greece" = "EL", "EU_27" = "EU_27")
  )

  # Filter to EU27
  df <- filter_eu27(df)
  df <- df[, c("geo", "time", "value")]

  # Convert quarters to dates (first day of quarter month)
  df$time <- gsub("Q1", "01", df$time)
  df$time <- gsub("Q2", "04", df$time)
  df$time <- gsub("Q3", "07", df$time)
  df$time <- gsub("Q4", "10", df$time)
  df$time <- floor_date(ym(df$time), "month")

  df$selection <- "gdp_quarterly:MillionEUR"

  # Reshape with time instead of year
  wide_df <- as.data.frame(
    acast(geo + time ~ selection, value.var = "value", data = df)
  )
  wide_df <- cbind(rownames(wide_df), rownames(wide_df), wide_df)
  colnames(wide_df)[1:2] <- c("geo", "time")
  wide_df$geo <- gsub("_.*", "", wide_df$geo)
  wide_df$time <- gsub(".*_", "", wide_df$time)
  rownames(wide_df) <- NULL

  save_processed(wide_df, "gdp_quarterly.csv")

  return(wide_df)
}


# -----------------------------------------------------------------------------
# GHG Intensity of Electricity Production (EEA)
# -----------------------------------------------------------------------------

process_ghg_intensity <- function() {
  message("\n--- Processing GHG Intensity of Electricity ---")

  df <- read_csv(
    file.path(RAW_DIR, "ghg_intensity_electricity_production.csv"),
    col_types = cols(
      `Greenhouse gas (GHG) emission intensity:number...3` = col_skip()
    ),
    show_col_types = FALSE
  )

  colnames(df)[1:3] <- c("year", "geo", "value")
  df <- df[-c(1, 2), 1:3]  # Remove header rows

  # Standardize country codes
  df$geo[df$geo == "EU-27"] <- "EU_27"
  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c("EU_27" = "EU_27")
  )

  df <- filter_eu27(df)
  df$selection <- "ghg_intensity_electricity_production"

  result <- reshape_to_wide(df[, c("selection", "year", "geo", "value")])

  save_processed(result, "ghg_intensity_electricity_production.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Heating and Cooling Degree Days (Eurostat)
# -----------------------------------------------------------------------------

process_hdd_cdd <- function() {
  message("\n--- Processing HDD/CDD ---")

  df <- read_csv(
    file.path(RAW_DIR, "hdd_cdd.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  colnames(df) <- c("unit", "selection", "geo", "time", "value")

  df <- filter_eu27(df)
  df <- df[, -1]  # Remove unit column

  # Convert to date format
  df$time <- floor_date(ym(df$time), "month")
  df$selection <- paste0("degree_days:", df$selection)

  # Reshape with time
  wide_df <- as.data.frame(
    acast(geo + time ~ selection, value.var = "value", data = df)
  )
  wide_df <- cbind(rownames(wide_df), rownames(wide_df), wide_df)
  colnames(wide_df)[1:2] <- c("geo", "time")
  wide_df$geo <- gsub("_.*", "", wide_df$geo)
  wide_df$time <- gsub(".*_", "", wide_df$time)
  rownames(wide_df) <- NULL

  save_processed(wide_df, "Monthly_hdd_cdd.csv")

  return(wide_df)
}


# -----------------------------------------------------------------------------
# Heat Pumps (Eurostat)
# -----------------------------------------------------------------------------

process_heat_pumps <- function() {
  message("\n--- Processing Heat Pumps ---")

  df <- read_csv(
    file.path(RAW_DIR, "heat_pumps_select.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  df <- filter_eu27(df)
  df <- df[, c("geo", "TIME_PERIOD", "OBS_VALUE")]
  colnames(df) <- c("geo", "year", "value")

  df$selection <- "heat_pumps:GWH"

  result <- reshape_to_wide(df)

  save_processed(result, "heat_pumps.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Land Use (FAO)
# -----------------------------------------------------------------------------

process_land_use <- function() {
  message("\n--- Processing Land Use ---")

  df <- read_csv(
    file.path(RAW_DIR, "LandUse.csv"),
    col_types = cols(
      `Area Code` = col_skip(),
      `Area Code (M49)` = col_skip(),
      `Item Code` = col_skip(),
      `Element Code` = col_skip()
    ),
    show_col_types = FALSE
  )

  # Remove flag columns (ending in F or N)
  flag_cols <- grep("[FN]$", colnames(df))
  if (length(flag_cols) > 0) {
    df <- df[, -flag_cols]
  }

  # Clean year column names (remove "Y" prefix)
  colnames(df) <- gsub("^Y", "", colnames(df))
  colnames(df)[1] <- "geo"

  # Convert to ISO2 codes
  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c(
      "Belgium-Luxembourg" = NA,
      "Channel Islands" = NA,
      "Czechoslovakia" = NA,
      "Serbia and Montenegro" = NA,
      "Yugoslav SFR" = NA
    )
  )

  df <- filter_eu27(df)

  # Create selection column
  df$selection <- paste0(
    "land_use:",
    gsub(" ", "_", df$Item), ":",
    gsub(" ", "_", df$Element), ":",
    gsub(" ", "_", df$Unit)
  )

  # Remove metadata columns and reshape
  df <- df[, !(colnames(df) %in% c("Item", "Element", "Unit"))]

  df_long <- melt(
    setDT(df),
    id.vars = c("geo", "selection"),
    variable.name = "year"
  )
  df_long <- setDF(df_long)
  df_long <- df_long[!is.na(df_long$value), ]

  result <- reshape_to_wide(df_long)

  save_processed(result, "land_use.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Modal Split Transport (Eurostat)
# -----------------------------------------------------------------------------

process_modal_split <- function() {
  message("\n--- Processing Modal Split Transport ---")

  df <- read_csv(
    file.path(RAW_DIR, "modal_split_transport.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  df <- filter_eu27(df)
  df <- df[, -2]  # Remove unit column

  colnames(df) <- c("selection", "geo", "year", "value")
  df$selection <- paste0("modal_split_transport:", df$selection)

  result <- reshape_to_wide(df)

  # Replace NA with 0 for modal split data
  result[is.na(result)] <- 0

  save_processed(result, "modal_split_transport.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Monthly Electricity Statistics (IEA)
# -----------------------------------------------------------------------------

process_electricity_stats <- function() {
  message("\n--- Processing Monthly Electricity Statistics ---")

  df <- read_csv(
    file.path(RAW_DIR, "Monthly_electricity_statistics.csv"),
    col_types = cols(Unit = col_skip()),
    skip = 8,
    show_col_types = FALSE
  )

  colnames(df) <- c("geo", "time", "balance", "product", "value")

  # Standardize country names
  df$geo <- mapvalues(
    df$geo,
    from = c("Republic of Turkiye", "North Macedonia",
             "People's Republic of China"),
    to = c("Turkey", "Macedonia", "China"),
    warn_missing = FALSE
  )

  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c(
      "EU_27" = "EU_27", "IEA Total" = "IEA Total",
      "OECD Americas" = "OECD Americas",
      "OECD Asia Oceania" = "OECD Asia Oceania",
      "OECD Europe" = "OECD Europe", "OECD Total" = "OECD Total"
    )
  )

  df <- filter_eu27(df)

  # Parse time strings (e.g., "January 2020" -> date)
  months <- c("January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December")
  years <- 2010:2023

  for (m in seq_along(months)) {
    for (y in years) {
      old_val <- paste(months[m], y)
      new_val <- paste0(sprintf("%02d", m), "-13-", y)
      df$time <- gsub(old_val, new_val, df$time, fixed = TRUE)
    }
  }

  df$time <- floor_date(mdy(df$time), "month")

  # Create selection column
  df$selection <- paste0(
    "Monthly_electricity_statistics:",
    df$balance, ":",
    df$product
  )

  df <- df[, c("selection", "geo", "time", "value")]

  # Reshape with time
  wide_df <- as.data.frame(
    acast(geo + time ~ selection, value.var = "value", data = df)
  )
  wide_df <- cbind(rownames(wide_df), rownames(wide_df), wide_df)
  colnames(wide_df)[1:2] <- c("geo", "time")
  wide_df$geo <- gsub("_.*", "", wide_df$geo)
  wide_df$time <- gsub(".*_", "", wide_df$time)
  rownames(wide_df) <- NULL

  # Remove remarks column if present
  remarks_col <- grep("Remarks", colnames(wide_df))
  if (length(remarks_col) > 0) {
    wide_df <- wide_df[, -remarks_col]
  }

  save_processed(wide_df, "Monthly_electricity_statistics.csv")

  return(wide_df)
}


# -----------------------------------------------------------------------------
# Monthly Oil Price Statistics (IEA)
# -----------------------------------------------------------------------------

process_oil_prices <- function() {
  message("\n--- Processing Monthly Oil Prices ---")

  df <- read_excel(
    file.path(RAW_DIR, "Monthly_oil_price_statistics.xlsx"),
    sheet = "raw data"
  )

  colnames(df) <- c("geo", "product", "flow", "unit", "time", "value")

  df$time <- floor_date(ymd(df$time), "month")

  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c("EU_27" = "EU_27")
  )

  df <- filter_eu27(df)

  # Create selection column
  df$selection <- paste0(
    "Monthly_oil_price_statistics:",
    df$product, ":",
    df$flow, ":",
    df$unit
  )

  df <- df[, c("selection", "geo", "time", "value")]

  # Reshape with time
  wide_df <- as.data.frame(
    acast(geo + time ~ selection, value.var = "value", data = df)
  )
  wide_df <- cbind(rownames(wide_df), rownames(wide_df), wide_df)
  colnames(wide_df)[1:2] <- c("geo", "time")
  wide_df$geo <- gsub("_.*", "", wide_df$geo)
  wide_df$time <- gsub(".*_", "", wide_df$time)
  rownames(wide_df) <- NULL

  save_processed(wide_df, "Monthly_oil_price_statistics.csv")

  return(wide_df)
}


# -----------------------------------------------------------------------------
# Population (Eurostat)
# -----------------------------------------------------------------------------

process_population <- function() {
  message("\n--- Processing Population ---")

  df <- read_csv(
    file.path(RAW_DIR, "population.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  df <- filter_eu27(df)
  df <- df[, -1]  # Remove first column

  colnames(df) <- c("selection", "geo", "year", "value")
  df$selection <- paste0("population:", df$selection)

  result <- reshape_to_wide(df)

  save_processed(result, "population.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Crops and Livestock Production (FAO)
# -----------------------------------------------------------------------------

process_crops_livestock <- function() {
  message("\n--- Processing Crops and Livestock ---")

  df <- read_csv(
    file.path(RAW_DIR, "Production_Crops_Livestock.csv"),
    col_types = cols(
      `Area Code` = col_skip(),
      `Area Code (M49)` = col_skip(),
      `Item Code` = col_skip(),
      `Item Code (CPC)` = col_skip(),
      `Element Code` = col_skip()
    ),
    show_col_types = FALSE
  )

  # Remove flag columns
  flag_cols <- grep("[FN]$", colnames(df))
  if (length(flag_cols) > 0) {
    df <- df[, -flag_cols]
  }

  # Clean column names
  colnames(df) <- gsub("^Y", "", colnames(df))
  colnames(df)[1] <- "geo"

  # Convert to ISO2
  df$geo <- countrycode(
    df$geo, "country.name", "iso2c",
    custom_match = c(
      "Belgium-Luxembourg" = NA,
      "Channel Islands" = NA,
      "Czechoslovakia" = NA,
      "Serbia and Montenegro" = NA,
      "Yugoslav SFR" = NA
    )
  )

  df <- filter_eu27(df)

  # Create selection column
  df$selection <- paste0(
    "crops_livestock:",
    gsub(" ", "_", df$Item), ":",
    gsub(" ", "_", df$Element), ":",
    gsub(" ", "_", df$Unit)
  )

  # Remove metadata columns
  df <- df[, !(colnames(df) %in% c("Item", "Element", "Unit"))]

  df_long <- melt(
    setDT(df),
    id.vars = c("geo", "selection"),
    variable.name = "year"
  )
  df_long <- setDF(df_long)

  result <- reshape_to_wide(df_long)

  save_processed(result, "crops_livestock.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Renewable Energy Share (Eurostat)
# -----------------------------------------------------------------------------

process_renewable_share <- function() {
  message("\n--- Processing Renewable Share ---")

  df <- read_csv(
    file.path(RAW_DIR, "share_of_renewable.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  df <- filter_eu27(df)
  df <- df[, -2]  # Remove unit column

  colnames(df) <- c("selection", "geo", "year", "value")
  df$selection <- paste0("renewable_share:", df$selection)

  result <- reshape_to_wide(df)

  save_processed(result, "renewable_share.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Solar Thermal Surface (Eurostat)
# -----------------------------------------------------------------------------

process_solar_thermal <- function() {
  message("\n--- Processing Solar Thermal Surface ---")

  df <- read_csv(
    file.path(RAW_DIR, "solar_thermal_surface.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  df <- filter_eu27(df)
  df <- df[, c("geo", "TIME_PERIOD", "OBS_VALUE")]
  colnames(df) <- c("geo", "year", "value")

  df$selection <- "solar_thermal_surface:THS_M2"

  result <- reshape_to_wide(df)

  save_processed(result, "solar_thermal_surface.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Trade Data (Eurostat)
# -----------------------------------------------------------------------------

process_trade <- function() {
  message("\n--- Processing Trade Data ---")

  df <- read_csv(
    file.path(RAW_DIR, "trade.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  # Map indicator codes to readable names
  df$indic_et <- mapvalues(
    df$indic_et,
    from = c("IVOL_EXP", "IVOL_IMP", "IVU_EXP", "IVU_IMP", "MIO_BAL_VAL",
             "MIO_EXP_VAL", "MIO_IMP_VAL", "PC_EXP_EU", "PC_IMP_EU",
             "RT_IVOL", "RT_IVU"),
    to = c("export_volume_index", "import_volume_index", "export_units",
           "import_units", "trade_balance_millionEUR", "export_millionEUR",
           "import_millionEUR", "share_export_toEU_perc", "share_import_toEU_perc",
           "volume_ratio_exportimport", "terms_of_trade_exportimport"),
    warn_missing = FALSE
  )

  # Map SITC codes to readable names
  df$sitc06 <- mapvalues(
    df$sitc06,
    from = c("SITC0_1", "SITC2_4", "SITC3", "SITC5", "SITC6_8",
             "SITC7", "TOTAL", "SITC9"),
    to = c("Food_drinks_tobacco", "Raw_materials", "Mineral_fuels_lubrificants",
           "Chemicals", "Other", "Machinery_transportequipment", "Total",
           "Not_Classified"),
    warn_missing = FALSE
  )

  df <- filter_eu27(df)

  df$selection <- paste0("trade:", df$indic_et, ":", df$sitc06, ":", df$partner)
  df <- df[, c("geo", "TIME_PERIOD", "OBS_VALUE", "selection")]
  colnames(df) <- c("geo", "year", "value", "selection")

  result <- reshape_to_wide(df)

  save_processed(result, "trade.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Train Performance (Eurostat)
# -----------------------------------------------------------------------------

process_train_performance <- function() {
  message("\n--- Processing Train Performance ---")

  df <- read_csv(
    file.path(RAW_DIR, "train_performance.csv"),
    col_types = cols(
      DATAFLOW = col_skip(),
      `LAST UPDATE` = col_skip(),
      freq = col_skip(),
      OBS_FLAG = col_skip()
    ),
    show_col_types = FALSE
  )

  # Map codes to readable names
  df$train <- mapvalues(
    df$train,
    from = c("TOTAL", "TRN_GD", "TRN_OTH", "TRN_PAS"),
    to = c("Total", "Goods_trains", "Other_trains", "Passenger_trains"),
    warn_missing = FALSE
  )

  df$vehicle <- mapvalues(
    df$vehicle,
    from = c("LOC", "RCA", "TOTAL"),
    to = c("Locomotives", "Railcars", "Total"),
    warn_missing = FALSE
  )

  df$mot_nrg <- mapvalues(
    df$mot_nrg,
    from = c("DIE", "ELC", "TOTAL"),
    to = c("Diesel", "Electricity", "Total"),
    warn_missing = FALSE
  )

  df$selection <- paste0(
    "train_performance:",
    df$train, ":",
    df$vehicle, ":",
    df$mot_nrg,
    ":THS_train_mk"
  )

  df <- filter_eu27(df)
  df <- df[, c("geo", "TIME_PERIOD", "OBS_VALUE", "selection")]
  colnames(df) <- c("geo", "year", "value", "selection")

  result <- reshape_to_wide(df)

  save_processed(result, "train_performance.csv")

  return(result)
}


# -----------------------------------------------------------------------------
# Climate Data (Copernicus ERA5)
# -----------------------------------------------------------------------------

process_climate <- function() {
  message("\n--- Processing Climate Data ---")

  # Population-weighted climate data
  climate_pop <- read_csv(
    file.path(RAW_DIR, "data_climate_country_pop_weight_1950_2023_updated.csv"),
    show_col_types = FALSE
  )

  climate_pop <- climate_pop[, c(2, 1, 7, 10, 12)]
  colnames(climate_pop) <- c(
    "geo", "year",
    "climate:rainfall:POP",
    "climate:temperature:POP",
    "climate:temperature_variability:POP"
  )

  climate_pop$geo <- countrycode(
    climate_pop$geo, "iso3c", "iso2c",
    custom_match = c(
      "CPT" = NA, "XB" = NA, "XC" = NA, "XD" = NA, "XE" = NA,
      "XF" = NA, "XG" = NA, "XH" = NA, "XI" = NA, "XL" = NA,
      "XM" = NA, "XO" = NA, "XU" = NA, "XV" = NA
    )
  )

  # Area-weighted climate data
  climate_area <- read_csv(
    file.path(RAW_DIR, "data_climate_country_1950_2023_updated.csv"),
    show_col_types = FALSE
  )

  climate_area <- climate_area[, c(2, 1, 7, 10, 12)]
  colnames(climate_area) <- c(
    "geo", "year",
    "climate:rainfall:AREA",
    "climate:temperature:AREA",
    "climate:temperature_variability:AREA"
  )

  climate_area$geo <- countrycode(
    climate_area$geo, "iso3c", "iso2c",
    custom_match = c(
      "CPT" = NA, "XB" = NA, "XC" = NA, "XD" = NA, "XE" = NA,
      "XF" = NA, "XG" = NA, "XH" = NA, "XI" = NA, "XL" = NA,
      "XM" = NA, "XO" = NA, "XU" = NA, "XV" = NA
    )
  )

  # Merge population and area weighted data
  climate_merged <- merge(climate_pop, climate_area, by = c("geo", "year"),
                          all = TRUE)
  climate_merged <- filter_eu27(climate_merged)

  save_processed(climate_merged, "climate.csv")

  return(climate_merged)
}


# =============================================================================
# Main Execution
# =============================================================================

main <- function() {
  message("=" |> rep(70) |> paste(collapse = ""))
  message("HISTORICAL DATA PREPROCESSING")
  message("=" |> rep(70) |> paste(collapse = ""))
  message(sprintf("Output directory: %s", OUTPUT_DIR))
  message(sprintf("Processing %d EU countries", length(EU27_COUNTRIES)))

  start_time <- Sys.time()

  # Process all data sources

  tryCatch(process_air_emissions(), error = function(e) message("Error: ", e$message))
  tryCatch(process_carbon_prices(), error = function(e) message("Error: ", e$message))
  tryCatch(process_energy_consumption(), error = function(e) message("Error: ", e$message))
  tryCatch(process_energy_taxes(), error = function(e) message("Error: ", e$message))
  tryCatch(process_ev_data(), error = function(e) message("Error: ", e$message))
  tryCatch(process_gdp_quarterly(), error = function(e) message("Error: ", e$message))
  tryCatch(process_ghg_intensity(), error = function(e) message("Error: ", e$message))
  tryCatch(process_hdd_cdd(), error = function(e) message("Error: ", e$message))
  tryCatch(process_heat_pumps(), error = function(e) message("Error: ", e$message))
  tryCatch(process_land_use(), error = function(e) message("Error: ", e$message))
  tryCatch(process_modal_split(), error = function(e) message("Error: ", e$message))
  tryCatch(process_electricity_stats(), error = function(e) message("Error: ", e$message))
  tryCatch(process_oil_prices(), error = function(e) message("Error: ", e$message))
  tryCatch(process_population(), error = function(e) message("Error: ", e$message))
  tryCatch(process_crops_livestock(), error = function(e) message("Error: ", e$message))
  tryCatch(process_renewable_share(), error = function(e) message("Error: ", e$message))
  tryCatch(process_solar_thermal(), error = function(e) message("Error: ", e$message))
  tryCatch(process_trade(), error = function(e) message("Error: ", e$message))
  tryCatch(process_train_performance(), error = function(e) message("Error: ", e$message))
  tryCatch(process_climate(), error = function(e) message("Error: ", e$message))

  end_time <- Sys.time()
  elapsed <- difftime(end_time, start_time, units = "secs")

  message("\n" |> paste0(rep("=", 70) |> paste(collapse = "")))
  message(sprintf("PREPROCESSING COMPLETE (%.1f seconds)", elapsed))
  message("=" |> rep(70) |> paste(collapse = ""))
}


# Run if executed directly
if (!interactive()) {
  main()
}