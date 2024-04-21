"""
Isabella Fisch, Caroline Han, Annie Zhou
DS 4400 - Machine Learning and Data Mining 1
Final Project
April 19, 2024
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def scale_series(series):
    """ standardizes a column from a dataframe
    :param series: pandas series
    :return: new_series: standardized series
    """
    avg = series.mean()
    stdev = series.std()
    values = []
    for x in series:
        x = (x - avg) / stdev
        values.append(x)
    new_series = values
    return new_series


def graph_feat_importance(model_obj, df, limit=None):
    """ plots a feature importance graph for a machine learning model
    :param model_obj: sklearn machine learning model
    :param df: dataframe the data is stored in
    :param limit: top number of features to include in the graph
    :return: None
    """
    # sort features in decreasing importance
    idx = np.argsort(model_obj.feature_importances_).astype(int)
    feat_list = [df.columns[_idx] for _idx in idx]
    feat_import = model_obj.feature_importances_[idx]

    if limit:
        # only take the top features
        feat_list = feat_list[-abs(limit):]
        feat_import = feat_import[-abs(limit):]

    # plot the graph
    plt.barh(feat_list, feat_import)
    plt.title('Feature Importance')
    plt.xlabel('Mean Increase in Gini')
    plt.ylabel('Features')
    plt.show()


def eval_reg_model(model_obj, train_x, train_y, test_x, test_y, df, train_error= True, linear=True,
                   feat_import_graph=False, limit=None):
    """ initiates, trains, and evaluates a regression model with given training data
    :param model_obj: sklearn machine learning model
    :param train_x: x training data
    :param train_y: y training data
    :param test_x: x testing data
    :param test_y: y testing data
    :param df: dataframe the data is stored in
    :param train_error: boolean indicating if the training error should be returned
    :param linear: boolean indicating if the model is linear regression
    :param feat_import_graph: boolean indicating if a feature importance graph should be plotted
    :param limit: top number of features to include in the graph
    :returns: mse, r2
    """
    # fit and predict using model
    model = model_obj
    model.fit(train_x, train_y)

    if train_error:
        train_preds = model.predict(train_x)
        train_mse = mean_squared_error(train_y, train_preds)
        train_r2 = r2_score(train_y, train_preds)
    else:
        train_mse = 'N/A'
        train_r2 = 'N/A'

    preds = model.predict(test_x)

    # get metrics
    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)

    if linear:
        # find each feature's coefficient and sort from biggest to smallest
        feat_coef = []
        for feat in range(len(model.coef_)):
            feat_coef.append((abs(model.coef_[feat]), train_x.columns[feat]))
        feat_coef.sort(reverse=True)
    else:
        feat_coef = 'N/A'

    if feat_import_graph:
        # graph feature importance
        graph_feat_importance(model, df, limit)

    # report results
    print(model_obj, '\nMean Squared Error:', round(mse, 3), '\nr2:', round(r2, 3), '\nTop 3 Important Features:',
          feat_coef[:3], '\n')
    return mse, r2, train_mse, train_r2


def main():

    # read in the data
    food = pd.read_csv('archive/train.csv')

    # identify and drop null values
    food = food.replace(to_replace=['NaN '], value=np.nan)
    food = food.dropna()

    ####################################################################################################

    # feature engineering
    # drop unnecessary columns
    food = food.drop(columns=['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Order_picked'])

    # calculate euclidean distance for restaurant and delivery location
    food['euclidean_distance'] = (((food['Restaurant_longitude'] - food['Delivery_location_longitude']) ** 2) +
                                  (food['Restaurant_latitude'] - food['Delivery_location_latitude']) ** 2) ** 0.5
    food = food.drop(columns=['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude',
                              'Delivery_location_longitude'])

    # get dummies - weather conditions, road traffic density, type of order, type of vehicle, city
    weather_dummies = pd.get_dummies(food['Weatherconditions'], prefix='Weather')
    traffic_dummies = pd.get_dummies(food['Road_traffic_density'], prefix='Traffic')
    order_dummies = pd.get_dummies(food['Type_of_order'], prefix='Order')
    vehicle_dummies = pd.get_dummies(food['Type_of_vehicle'], prefix='Vehicle')
    city_dummies = pd.get_dummies(food['City'], prefix='City')

    # concatenate dummies to food df and drop the columns with categorical values
    food = pd.concat([food, weather_dummies, traffic_dummies, order_dummies, vehicle_dummies, city_dummies], axis=1)
    food = food.drop(columns=['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'City'])

    # only keep the hour for time ordered
    food_hour = []
    for row in food['Time_Ordered']:
        hour = str(row).split(':')[0]
        food_hour.append(int(hour))
    food['Hour'] = food_hour
    food = food.drop(columns=['Time_Ordered'])

    # convert multiple deliveries, festival to binary values
    food['multiple_deliveries'] = food['multiple_deliveries'].apply(lambda x: 1 if x == 'yes' else 0)
    food['Festival'] = food['Festival'].apply(lambda x: 1 if x == 'yes' else 0)

    # separate target value from rest of data
    food_y = food['Time_taken(min)']
    food_y_lst = []
    for row in food_y:
        mins = row.split()[1]
        food_y_lst.append(int(mins))
    food_y = food_y_lst
    food = food.drop(columns=['Time_taken(min)'])

    # convert data types
    food['Delivery_person_Age'] = food['Delivery_person_Age'].astype(int)
    food['Delivery_person_Ratings'] = food['Delivery_person_Ratings'].astype(float)

    # standardize numerical columns
    food['Delivery_person_Age'] = scale_series(food['Delivery_person_Age'])
    food['Delivery_person_Ratings'] = scale_series(food['Delivery_person_Ratings'])

    # split data into training and testing
    food_x_train, food_x_test, food_y_train, food_y_test = train_test_split(food, food_y, test_size=0.25,
                                                                            random_state=22)

    ####################################################################################################

    # demographic evaluation
    # read in Indian province data and merge with demographic data
    states = gpd.read_file("ne_50m_admin_1_states_provinces.zip")
    demographics = pd.read_csv('india.csv')
    india = states[states['iso_a2'] == 'IN']
    india = pd.merge(india, demographics, left_on='name', right_on='States_Union Territories')

    # map coordinates from food data to Indian state data
    coord = food.copy()
    coord = gpd.GeoDataFrame(
        coord, geometry=gpd.points_from_xy(coord.Delivery_location_longitude, coord.Delivery_location_latitude),
        crs="EPSG:4326"
    )
    coord = coord.drop(coord[coord['Delivery_location_longitude'] < 60].index)

    # make plots to visualize delivery locations vs population, unemployment rate, and poverty rate
    coord = coord.to_crs(india.crs)
    base = india.plot(column="2011- POP", legend=True,
                      legend_kwds={"label": "Population"})
    coord.plot(ax=base, color='red', markersize=1)
    plt.title('Delivery Locations vs. Population')

    base = india.plot(column="2011 -UNEMP", legend=True,
                      legend_kwds={"label": "Unemployment Rate"})
    coord.plot(ax=base, color='red', markersize=1)
    plt.title('Delivery Locations vs. Unemployment Rate')

    base = india.plot(column="2011 -Poverty", legend=True,
                      legend_kwds={"label": "Poverty Rate"})
    coord.plot(ax=base, color='red', markersize=1)
    plt.title('Delivery Locations vs. Poverty Rate')

    ####################################################################################################

    # training models
    # linear regression
    lin_mse, lin_r2, lin_mse_train, lin_r2_train = eval_reg_model(LinearRegression(), food_x_train, food_y_train,
                                                                  food_x_test, food_y_test, food)

    # ridge regression - try different lambda values
    alphas = [0.5, 1, 10, 100, 1000]
    ridge_errors = []
    ridge_r2s = []
    for alpha in alphas:
        ridge_mse, ridge_r2, ridge_mse_train, ridge_r2_train = eval_reg_model(Ridge(alpha=alpha), food_x_train,
                                                                              food_y_train, food_x_test,
                                                                              food_y_test, food)
        ridge_errors.append(ridge_mse)
        ridge_r2s.append(ridge_r2)

    # make a plot displaying how mse and r2 change with lambda
    fig, ax_mse = plt.subplots()
    ax_mse.set_xlabel('Lambda Values')
    ax_mse.set_ylabel('Mean Squared Error', color='red')
    ax_mse.plot([str(a) for a in alphas], ridge_errors, color='red')
    ax_mse.tick_params(axis='y', labelcolor='red')
    ax_r2 = ax_mse.twinx()
    ax_r2.set_ylabel('r2', color='blue')
    ax_r2.plot([str(a) for a in alphas], ridge_r2s, color='blue')
    ax_r2.tick_params(axis='y', labelcolor='blue')
    plt.title('Metrics for Ridge Regression as Lambda Increases')
    plt.show()

    # decision tree - try different max depths
    tree_errors = []
    tree_errors_train = []
    tree_r2s = []
    tree_r2s_train = []
    for depth in range(1, 11):
        tree_mse, tree_r2, tree_mse_train, tree_r2_train = eval_reg_model(DecisionTreeRegressor(max_depth=depth),
                                                                          food_x_train, food_y_train, food_x_test,
                                                                          food_y_test, food, linear=False)
        tree_errors.append(tree_mse)
        tree_r2s.append(tree_r2)
        tree_errors_train.append(tree_mse_train)
        tree_r2s_train.append(tree_r2_train)

    # make a plot displaying how mse and r2 change with tree depth
    fig, ax_mse = plt.subplots()
    ax_mse.set_xlabel('Tree Depths')
    ax_mse.set_ylabel('Mean Squared Error', color='red')
    ax_mse.plot([depth for depth in range(1, 11)], tree_errors, color='red')
    ax_mse.tick_params(axis='y', labelcolor='red')
    ax_r2 = ax_mse.twinx()
    ax_r2.set_ylabel('r2', color='blue')
    ax_r2.plot([depth for depth in range(1, 11)], tree_r2s, color='blue')
    ax_r2.tick_params(axis='y', labelcolor='blue')
    plt.title('Metrics for Decision Tree Regressor as Max Depth Increases')
    plt.show()

    # make a plot showing how error changes with tree depth
    plt.plot(range(1, 11), tree_errors_train, label='Training Error', color='blue')
    plt.plot(range(1, 11), tree_errors, label='Testing Error', color='red')
    plt.title('Error as Tree Depth Increases')
    plt.xlabel('Tree Depth')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    # plot a visualization for the optimal tree
    tree_vis = DecisionTreeRegressor(max_depth=5)
    tree_vis.fit(food_x_train, food_y_train)
    fig = plt.figure(figsize=(50, 50))
    _ = plot_tree(tree_vis, feature_names=food.columns)

    # random forest - try different max depths and number of estimators
    num_estimators = [10, 50, 100, 500]
    forest_errors = []
    forest_r2s = []
    forest_table = pd.DataFrame(columns=['Estimators', 'Depth', 'MSE', 'r2'])
    index = 0
    for num in num_estimators:
        for depth in range(1, 11):
            forest_mse, forest_r2, forest_mse_train, forest_r2_train = eval_reg_model(RandomForestRegressor(n_estimators=num, max_depth=depth),
                                                                                      food_x_train, food_y_train,
                                                                                      food_x_test, food_y_test,
                                                                                      food, linear=False,
                                                                                      feat_import_graph=True)
            forest_errors.append(forest_mse)
            forest_r2s.append(forest_r2)
            forest_table.loc[index] = {'Estimators': num, 'Depth': depth, 'MSE': forest_mse, 'r2': forest_r2}
            index += 1
    print(forest_table)


main()
