#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[6]:


def reduce_mem_usage(df, verbose=True):
    """
    Function to reduce the required memory of a dataframe.
    Returns the dataframe with reduced size.
    """
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Prepare the data

# In[7]:


# Load pre-processed data and drop NAs
df_train_validation = pd.read_csv("final_train_val.csv", low_memory=False, index_col="id")
df_train_validation = df_train_validation.dropna()
df_test = pd.read_csv("final_test.csv", low_memory=False, index_col="id")
df_test = df_test.dropna()


# In[8]:


# Reduce size of dataframes
df_train_validation = reduce_mem_usage(df_train_validation)
df_test = reduce_mem_usage(df_test)


# In[9]:


# Get series of origin and destination airports
orig_airports = pd.Series(df_train_validation['ORIGIN_AIRPORT'])
dest_airports = pd.Series(df_train_validation['DESTINATION_AIRPORT'])

# Get a list of all unique airports and create an ID mapping
all_airports = orig_airports.append(dest_airports).unique()
airport_ids = pd.Series(list(range(len(all_airports))), index=all_airports)
airport_id_dict = {airport_ids[a]: a for a in all_airports}

# Change from IATA codes to generated IDs
orig_airports = orig_airports.apply(lambda x: airport_ids[x])
df_train_validation["ORIGIN_AIRPORT"] = orig_airports
dest_airports = dest_airports.apply(lambda x: airport_ids[x])
df_train_validation["DESTINATION_AIRPORT"] = dest_airports
dist_airports = pd.Series(df_train_validation['scaled_DISTANCE'])


# In[10]:


# Split the data into train, validation, and test sets
df_train, df_validation = train_test_split(df_train_validation, test_size=0.20, random_state = 42)
X_train, y_train = df_train.drop("ARRIVAL_DELAY", axis=1), df_train["ARRIVAL_DELAY"]
X_val, y_val = df_validation.drop("ARRIVAL_DELAY", axis=1), df_validation["ARRIVAL_DELAY"]
X_test = df_test


# ## Airport features

# In[11]:


# Because the destination and origin airports are the same 
assert(set(df_train_validation["DESTINATION_AIRPORT"].unique()) == set(df_train_validation["ORIGIN_AIRPORT"].unique()))
# we can use only origin airports
node_data = df_train_validation[["ORIGIN_AIRPORT", "LATITUDE_origin", "LONGITUDE_origin"]]
node_data = node_data.drop_duplicates(ignore_index=True).values[:,1:3]


# ## Edge features

# In[12]:


# Extract edge embedding
edge_data = df_train_validation[['MONTH', 'DAY', 'DAY_OF_WEEK', 'FLIGHT_NUMBER', 'SCHEDULED_DEPARTURE', 'TAXI_OUT', 'SCHEDULED_ARRIVAL', 'ARRIVAL_DELAY', 
                     'DEPARTURE_DELAY', 'AIRLINE_AA', 'AIRLINE_AS', 'AIRLINE_B6', 'AIRLINE_DL', 'AIRLINE_EV', 'AIRLINE_F9', 'AIRLINE_HA', 'AIRLINE_MQ',
                     'AIRLINE_DL', 'AIRLINE_EV', 'AIRLINE_F9', 'AIRLINE_HA', 'AIRLINE_MQ', 'AIRLINE_NK', 'AIRLINE_OO', 'AIRLINE_UA', 'AIRLINE_US', 'AIRLINE_VX',
                     'AIRLINE_WN', 'scaled_DEPARTURE_TIME', 'scaled_WHEELS_OFF', 'scaled_SCHEDULED_TIME']].values
# Extract edge labels, i.e. target
edge_labels = df_train_validation['ARRIVAL_DELAY'].values


# # Graph learning

# In[13]:


import numpy as np
import torch
import dgl
from torch import nn
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


# In[14]:


# Build graph
src = orig_airports.to_numpy()
dst = dest_airports.to_numpy()
edge_pred_graph = dgl.graph((src,dst))
edge_pred_graph.ndata['feature'] = torch.from_numpy(node_data)
edge_pred_graph.edata['feature'] = torch.from_numpy(edge_data)
edge_pred_graph.edata['label'] = torch.from_numpy(edge_labels.reshape(-1,1))
edge_pred_graph.edata['train_mask'] = torch.zeros(len(df_train_validation), dtype=torch.bool)
edge_pred_graph.edata['train_mask'][0:len(df_train)] = 1


# In[15]:


import dgl.function as fn
class DotProductPredictor(nn.Module):
    """
    Applies the dot product between all edges to compute the labels
    """
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


# In[16]:


class SAGE(nn.Module):
    """
    A two layer SAGEConv network
    """
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


# In[18]:


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()
    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


# In[35]:


# Extract features
node_features = edge_pred_graph.ndata['feature']
edge_label = edge_pred_graph.edata['label']
train_mask = edge_pred_graph.edata['train_mask']
model = Model(2, 100, 1)
model = model.float()
opt = torch.optim.Adam(model.parameters(), lr=0.04)

# Training
for epoch in range(50):
    batch_mask = torch.zeros(train_mask.shape[0], dtype=torch.bool).bernoulli(0.4)
    pred = model(edge_pred_graph, node_features.float())
    loss = ((pred[batch_mask] - edge_label[batch_mask]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print("Sample: ","Predicted:", pred[batch_mask][0].tolist()[0], "Actual:", edge_label[batch_mask][0].tolist()[0])
        print(loss.item())


# In[36]:


pred

