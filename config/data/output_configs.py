output_configs = {}
output_configs['output'] = 'Sectors' # 'Total' 'TotalECON' 'TotalHOUSE' 'Sectors'

output_configs['emission_type'] = 'CO2' # 'GHG' 'CO2'
output_configs['measure'] = 'KG_HAB' # 'KG_HAB' 'THS_T' 'both'

output_configs['mode'] = 'level' # 'difference' 'level'

output_configs['grouping_structure'] = {
    'HeatingCooling': ['HeatingCoolingbyHousehold'],
    'Industry': ['C-Manufacturing', 'B-Mining', 'F-Construction'],
    'Land': ['A-Agricolture,Forestry,Fishing', 'E-Water,Waste'],
    'Mobility': ['H-Transportation,Storage', 'G-Trade,VehicleRepair', 'TransportbyHousehold'],
    'Power': ['D-Electricity,Gas,Steam,AirConditioning'],
    'Other': ['Q-Health', 'M-Science,Technical,Professional', 'K-Finance,Insurance', 'J-Information,Communication', 'R-Arts,Entertainement', 'S-OtherService', 'T-HouseholdasemployersActivities', 'O-PublicAdmin,Defence', 'OtherbyHousehold', 'U-ExtraterritorialOrgsActivities', 'N-Administrative', 'L-RealEstate', 'I-Accomodation,FoodService', 'P-Education'],
    'TotalHOUSE': 'TotalHousehold',
    'TotalECON': 'Total',
    'Total': 'AllNACEplusHousehold'
}
