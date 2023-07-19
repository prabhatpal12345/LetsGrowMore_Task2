#!/usr/bin/env python
# coding: utf-8

# # Position : Data Science Intern

# Task 2 :Stock Market Prediction And Forecasting Using Stacked LSTM

# In[38]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[39]:


import torch
import torch.nn as nn


# In[40]:


df = pd.read_csv("net.csv")
closed_prices = df["Close"]


# In[41]:


seq_len = 15


# In[42]:


mm = MinMaxScaler()
scaled_price = mm.fit_transform(np.array(closed_prices)[... , None]).squeeze()


# In[43]:


X = []
y = []


# In[44]:


for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i + seq_len])
    y.append(scaled_price[i + seq_len])


# In[45]:


X = np.array(X)[... , None]
y = np.array(y)[... , None]


# In[46]:


train_x = torch.from_numpy(X[:int(0.8 * X.shape[0])]).float()
train_y = torch.from_numpy(y[:int(0.8 * X.shape[0])]).float()
test_x = torch.from_numpy(X[int(0.8 * X.shape[0]):]).float()
test_y = torch.from_numpy(y[int(0.8 * X.shape[0]):]).float()


# In[47]:


class Model(nn.Module):
    def __init__(self , input_size , hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size , batch_first = True)
        self.fc = nn.Linear(hidden_size , 1)
    def forward(self , x):
        output , (hidden , cell) = self.lstm(x)
        return self.fc(hidden[-1 , :])
model = Model(1 , 64)


# In[48]:


optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELoss()


# In[49]:


num_epochs = 100


# In[ ]:


for epoch in range(num_epochs):
    output = model(train_x)
    loss = loss_fn(output , train_y)


# In[ ]:


optimizer.zero_grad()
   loss.backward()
   optimizer.step()


# In[ ]:


if epoch % 10 == 0 and epoch != 0:
        print(epoch , "epoch loss" , loss.detach().numpy())


# In[ ]:


model.eval()
with torch.no_grad():
    output = model(test_x)


# In[ ]:


pred = mm.inverse_transform(output.numpy())
real = mm.inverse_transform(test_y.numpy())


# In[ ]:


plt.plot(pred.squeeze() , color = "red" , label = "predicted")
plt.plot(real.squeeze() , color = "green" , label = "real")
plt.show()


# In[ ]:




