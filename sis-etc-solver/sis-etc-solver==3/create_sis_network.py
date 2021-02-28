import pandas as pd

# number of nodes (max: 50)
n = 50

# read network data
df_US_airport_ranking = pd.read_csv(
    './data/US_Airport_Ranking.csv').head(n)
df_US_airport_iata = pd.read_csv('./data/US_Airport_IATA.csv')
df_route = pd.read_csv('./data/route.csv')

# link Airport Name and IATA in df_US_airport_ranking
for index, row in df_US_airport_ranking.iterrows():
    df_US_airport_ranking.at[index, 'IATA'] = df_US_airport_iata[df_US_airport_iata['Airport Name']
                                                                 == row['Airport Name']]['IATA'].values[0]

# extract route data according to Source IATA
df_route_source = pd.DataFrame()
for index, row in df_US_airport_ranking.iterrows():
    df_route_source = pd.concat(
        [df_route_source, df_route[df_route["Source airport"] == str(row["IATA"])]])

# extract route data according to Destination IATA
df_route_source_dest = pd.DataFrame()
for index, row in df_US_airport_ranking.iterrows():
    df_route_source_dest = pd.concat(
        [df_route_source_dest, df_route_source[df_route_source["Destination airport"] == str(row["IATA"])]])

# create adjecent matrix according to flights
df_ad_matrix = pd.DataFrame(
    index=df_US_airport_ranking['IATA'].values, columns=df_US_airport_ranking['IATA'].values)
df_ad_matrix = df_ad_matrix.fillna(0)
for index, row in df_route_source_dest.iterrows():
    df_ad_matrix.at[row['Source airport'], row['Destination airport']] += 1

# create weight adjecent matrix according to passengers
for index, row in df_ad_matrix.iterrows():
    df_ad_matrix[index] = df_ad_matrix[index].values * \
        df_US_airport_ranking[df_US_airport_ranking['IATA']
                              == index]['Passenger'].values[0]

# normalization of adjecent matrix
df_ad_matrix = df_ad_matrix.T / df_ad_matrix.values.max()
df_ad_matrix.to_csv('./data/US_Airport_Ad_Matrix.csv')
