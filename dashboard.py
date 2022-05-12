import pandas as pd
import streamlit as st
import plotly.express as px
import geopandas
import folium

from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
from datetime import datetime

# Read data
@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )

    return data

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    try:
        geofile = geopandas.read_file( url )

        return geofile
    except:
        return None

@st.cache( allow_output_mutation=True )
def data_transform( data ):
    data['date'] = pd.to_datetime( data['date'] )
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def build_multiselect( msg, filter ):
    return st.sidebar.multiselect(
        msg,
        filter
    )

def data_overview( data ):
    f_attributes = build_multiselect( 'Enter columns', data.columns )

    f_zipcode = build_multiselect( 'Enter zipcode', data.zipcode.unique() )

    if (f_attributes != []) and (f_zipcode != []):
        data = data.loc[data['zipcode'].isin( f_zipcode ), f_attributes]
    elif (f_attributes != []) and (f_zipcode == []):
        data = data.loc[:, f_attributes]
    elif (f_attributes == []) and (f_zipcode != []):
        data = data.loc[data['zipcode'].isin( f_zipcode ), :]
    else:
        data = data.copy()

    st.dataframe( data.head() )

    return data

def average_metrics_by_region( data ):
    c1, c2 = st.columns( (1, 1) )
    m_r_columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/m2']

    df_grouped_region_1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()

    df_grouped_region_2 = data[['zipcode', 'price', 'sqft_living', 'price_m2']] \
        .groupby( 'zipcode' ).mean().reset_index()
    
    df_grouped_region = pd.merge(df_grouped_region_1, df_grouped_region_2, on='zipcode', how='inner')
    df_grouped_region.columns = m_r_columns

    c1.header( 'Average Values' )
    c1.dataframe( df_grouped_region, height=600 )

    # Descriptive Stats
    num_attributes = data.select_dtypes( include=['int64', 'float64'] )

    c2.header( 'Descriptive Analysis' )
    c2.dataframe( num_attributes.describe( percentiles=[] ).T.reset_index() \
        .rename(columns={"index":"attributes", "50%":"median"}) \
        .drop(['count', 'std'], axis=1), height=600 )

    return None

def price_density_maps( data, geofile ):
    st.title( 'Region Overview' )

    m1, m2 = st.columns( (1, 1) )

    # maps_df = data.copy()
    maps_df = data.sample( 100 )

    # Base Map - Folium
    density_map = folium.Map( location=[maps_df['lat'].mean(),
        maps_df['long'].mean()],
        default_zoom_start=15 )

    marker_cluster = MarkerCluster().add_to( density_map )

    for name, row in maps_df.iterrows():
        folium.Marker( [row['lat'], row['long']],
        popup=f"Price ${row['price']} on: {row['date']}  Features: {row['sqft_living']}" 
        + f"sqft, {row['bedrooms']} bedrooms, {row['bathrooms']}"
        + f" bathrooms, year built: {row['yr_built']}" ).add_to( marker_cluster )
    
    m1.header( 'Portfolio Density' )
    with m1:
        folium_static( density_map )

    # Region Price Map
    m2.header( 'Price Density' )

    df_m2 = maps_df[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df_m2.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin( df_m2['ZIP'].tolist() )]

    region_price_map = folium.Map( location=[data['lat'].mean(),
            data['long'].mean()],
            default_zoom_start=15 )
    
    region_price_map.choropleth( data = df_m2,
        geo_data = geofile,
        columns=['ZIP', 'PRICE'],
        key_on='feature.properties.ZIP',
        fill_color='YlOrRd',
        fill_opacity = 0.7,
        line_opacity = 0.2,
        legend_name = 'AVG PRICE' )

    with m2:
        folium_static( region_price_map )
    
    return None

def comercial_attributes_distributions( data ):
    st.sidebar.title( 'Commercial Options' )
    st.title( 'Commercial Attributes' )

    # Average Price per Year
    min_year_built = int( data['yr_built'].min() )
    max_year_built = int( data['yr_built'].max() )

    st.sidebar.subheader( 'Select Max Year Built' )
    f_year_built = st.sidebar.slider( 'Year Built', min_year_built, max_year_built, max_year_built )

    st.subheader( 'Average Price per Year Built' )
    df_prices_year = data.loc[data['yr_built'] <= f_year_built]
    df_prices_year = df_prices_year[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

    fig = px.line( df_prices_year, x='yr_built', y='price' )
    st.plotly_chart( fig, use_container_width=True )

    # Average Price per Day
    st.subheader( 'Average Price per Day' )
    st.sidebar.subheader( 'Select Max Date' )

    # filters
    min_date = datetime.strptime( data['date'].min().strftime( '%Y-%m-%d' ), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max().strftime( '%Y-%m-%d' ), '%Y-%m-%d' )

    f_date = st.sidebar.slider( 'Date', min_date, max_date, max_date )
    data['date'] = pd.to_datetime( data['date'] )
    df_prices_day = data[data['date'] <= f_date]
    df_prices_day = df_prices_day[['date', 'price']].groupby( 'date' ).mean().reset_index()

    fig = px.line( df_prices_day, x='date', y='price' )
    st.plotly_chart( fig, use_container_width=True )
    
    # Histograms
    st.header( 'Price Distribution' )
    st.sidebar.subheader( 'Select Max Price' )

    # filter
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

    df = data.loc[data['price'] <= f_price]

    # data plot
    fig = px.histogram ( df, x='price', nbins=50 )
    st.plotly_chart( fig, use_container_width=True )

    return None

def non_comercial_distributions( data ):
    st.sidebar.title( 'Attributes Options ')
    st.title( 'House Attributes' )

    # filters
    f_bedrooms = st.sidebar.selectbox( 'Max number of bedrooms', sorted( set( data['bedrooms'].unique() ), reverse=True) )
    f_bathrooms = st.sidebar.selectbox( 'Max number of bathrooms', sorted( set( data['bathrooms'].unique() ), reverse=True ) )

    c1, c2 = st.columns( 2 )
    # House per bedrooms
    c1.header( 'Houses per bedrooms' )
    df_dists = data[data['bedrooms'] <= f_bedrooms]
    fig = px.histogram( df_dists, x='bedrooms', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    # House per bathrooms
    c2.header( 'Houses per bathrooms' )
    df_dists =  data[data['bathrooms'] <= f_bathrooms]
    fig = px.histogram( df_dists, x='bathrooms', nbins=19 )
    c2.plotly_chart( fig, use_container_width=True )

    # filters
    f_floors = st.sidebar.selectbox( 'Max number of floors', sorted( set( data['floors'].unique() ), reverse=True ))
    f_waterview = st.sidebar.checkbox( 'Only Houses with Water View')

    c1, c2 = st.columns( 2 )

    # House per floors
    c1.header( 'Houses per floor')
    df_floors = data[data['floors'] < f_floors]

    # plot
    fig = px.histogram( df_floors, x='floors', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    # House per water view
    c2.header( 'Houses per Water View')
    if f_waterview:
        df_waterview = data[data['waterfront'] == 1]
    else:
        df_waterview = data.copy()

    # plot
    fig = px.histogram( df_waterview, x='waterfront', nbins=10 )
    c2.plotly_chart( fig, use_container_width=True )

    return None

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title( 'House Rocket' )

    # ETL

    ## Extraction
    # get data
    st.header( 'Data Overview' )

    path = f'./dataset/kc_house_data.csv'

    data = get_data( path )

    # get geofile
    url = "https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson"
    geofile = get_geofile( url )

    ## Transformation
    data = data_transform( data )

    ## Loading

    # Filtering Attributes
    data_overview( data )
    # data = data_overview( data )

    # Average Metrics by Region
    average_metrics_by_region( data )

    # Plotting Portfolio Density MAPS
    price_density_maps( data, geofile )

    # Plotting Comercial Attributes Distributions
    comercial_attributes_distributions( data )

    # Plotting Non-Commercial Attributes Distributions
    non_comercial_distributions( data )
