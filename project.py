import torch
import numpy as np
import pandas as pd
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F

from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)

# Constants:
DONT_INCLUDE = ['R_fighter', 'B_fighter']
CSV_saved = 'ufc-master.csv'
CSV = 'ufc-master.csv'
SMALL_N = 1e-12

"""
ReadCSV takes in a name for a csv, and creates
a dict in which (f = fighter) : key = (f1, f2)
value = [{}]. Value dict is key: feature, value: feature
value
"""
def ReadCSV(data_csv):
    df = pd.read_csv('ufc-master.csv')
    dict_4_data = {}
    for index, row in df.iterrows():
        # make the mapping (fighter1, fighter2) in alphabetical order,
        # and include date
        key = tuple(sorted((row['R_fighter'], row['B_fighter'])))
        value = {col: row[col] for col in df.columns if col not in DONT_INCLUDE}
        if key in dict_4_data:
            # add another value to dict_4_data
             dict_4_data[key].append(value)
        else:
            dict_4_data[key] = [value]
    return dict_4_data

"""
Helpers to store data to a textfile, and load it from
there, also function update to call to do actually
recreate data
"""
def store(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
def load(filename = 'storage.txt'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
def updateData():
    store(ReadCSV(CSV), CSV_saved)

"""
Prototype Baseline takes in data dict and then will
predict the winning fighter based on who has the higher win
rate. Returns loss
"""
def Baseline1(data, p):
    count_of_wrong = 0
    for fights, fights_data in data.items():
        # If there is more than one fight per pair of fighters
        # Greedly choose the most recent one
        fight = fights_data[-1]
        # red_fighter, blue_fighter = fights
        # Choose which fighter won
        red_fighter_won = fight['Winner'] == 'Red'
        # Find each fighters win rate
        red_win_rate = fight['R_wins'] / (fight['R_wins'] + fight['R_losses'] + fight['R_draw'] + SMALL_N)
        blue_win_rate = fight['B_wins'] / (fight['B_wins'] + fight['B_losses'] + fight['B_draw'] + SMALL_N)
        # With probability p choose the fighter with a better win rate
        # Choose the fighter with the better win rate with prob p
        if random.random() <= p:
            red_fighter_pred = red_win_rate >= blue_win_rate
        # If we choose wrong than add one to account for loss
        if red_fighter_won != red_fighter_pred:
            count_of_wrong += 1
    # return the loss
    return count_of_wrong / len(data)

"""
Prototype Baseline takes in data dict and then will
predict the winning fighter based on who actually won
with p chance
"""
def Baseline2(data, p):
    count_of_wrong = 0
    for fights, fights_data in data.items():
        # If there is more than one fight per pair of fighters
        # Greedly choose the most recent one
        fight = fights_data[-1]
        # red_fighter, blue_fighter = fights
        # Choose which fighter won
        red_fighter_won = fight['Winner'] == 'Red'
        # With probability p choose the fighter with a better win rate
        # Choose the fighter with the better win rate with prob p
        if random.random() <= p:
            red_fighter_pred = red_fighter_won
        else:
            red_fighter_pred = not red_fighter_pred
        # If we choose wrong than add one to account for loss
        if red_fighter_won != red_fighter_pred:
            count_of_wrong += 1
    # return the loss
    return count_of_wrong / len(data)

#data = load()
data = ReadCSV(CSV)
print(Baseline1(data, 1))
print(Baseline2(data, 1))
#example
#print(data[("Johnny Walker", "Thiago Santos")])


###################################################

x = pd.read_csv(CSV)
# Change the winner to 0 or 1: 1 = red won
for i, row in x.iterrows():
     if row['Winner'] == 'Red':
         x.loc[i, 'Winner'] = 1
     else:
         x.loc[i, 'Winner'] = 0
# We cant use strings since you know they are not numbers, so we will drop them
# drop_strings = ['R_fighter', 'B_fighter', 'Referee', 'date', 'location', 'weight_class',
#                 'B_Stance', 'R_Stance']
# We also dont want to use the following: These can be irrelevent columns or something
drop_other = ['location', 'date']
x = x.drop(columns=drop_other)
#Lets trun our bool values into numerical values
convert_bool = ['title_bout']
x[convert_bool] = x[convert_bool].astype(int)
# Fill in for missing data might consider just deleating incomplet data
x = x.fillna(0)

# testing out LabelEncoder:
from sklearn.preprocessing import LabelEncoder
col = x.columns
num_col = x._get_numeric_data().columns
a = list(set(col) - set(num_col))

# Convert categorical data to numerical
label = LabelEncoder()
for i in a:
    x[i] = x[i].astype(str)
    x[i] = label.fit_transform(x[i])

print(x['Winner'])
# Lets make our label whether red fighter won
winner = 'Winner'
# # Now we have a feat feature vector and a ys for the classes 1 = red won
# feat = x.drop(columns=[winner]).values
# ys = x[winner].values

###################################################

# ------------------------------------------MODEL--------------------------------------------
#Sources: https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6
# https://medium.com/@carlosrodrigo.coelho/logistic-regression-pytorch-956f96b28010
# split data

#This model of linear regression from
# Prepare data


# column_names = feat[0]
# print(column_names)
# data = feat[1:]
ufc = x.copy()
# Combine seperate data
ufc['odds_dif'] = ufc['R_odds']-ufc['B_odds']
ufc['ev_dif'] = ufc['R_ev']-ufc['B_ev']
ufc['current_lose_streak_dif'] = ufc['R_current_lose_streak']-ufc['B_current_lose_streak']
ufc['current_win_streak_dif' ] = ufc['R_current_win_streak']-ufc['B_current_win_streak']
ufc['draw_dif'] = ufc['R_draw']-ufc['B_draw']
ufc['avg_SIG_STR_landed_dif'] = ufc['R_avg_SIG_STR_landed']-ufc['B_avg_SIG_STR_landed']
ufc['avg_SIG_STR_pct_dif'] = ufc['R_avg_SIG_STR_pct']-ufc['B_avg_SIG_STR_pct']
ufc['avg_SUB_ATT_dif'] = ufc['R_avg_SUB_ATT']-ufc['B_avg_SUB_ATT']
ufc['avg_TD_landed_dif'] = ufc['R_avg_TD_landed']-ufc['B_avg_TD_landed']
ufc['avg_TD_pct_dif'] = ufc['R_avg_TD_pct']-ufc['B_avg_TD_pct']
ufc['longest_win_streak_dif' ] = ufc['R_longest_win_streak']-ufc['B_longest_win_streak']
ufc['losses_dif'] = ufc['R_losses']-ufc['B_losses']
ufc['total_rounds_fought_dif'] = ufc['R_total_rounds_fought']-ufc['B_total_rounds_fought']
ufc['total_title_bouts_dif'] = ufc['R_total_title_bouts']-ufc['B_total_title_bouts']
ufc['win_by_Decision_Majority_dif'] = ufc['R_win_by_Decision_Majority']-ufc['B_win_by_Decision_Majority']
ufc['win_by_Decision_Split_dif'] = ufc['R_win_by_Decision_Split']-ufc['B_win_by_Decision_Split']
ufc['win_by_Decision_Unanimous_dif'] = ufc['R_win_by_Decision_Unanimous']-ufc['B_win_by_Decision_Unanimous']
ufc['Win_by_KO/TKO_dif'] = ufc['R_win_by_KO/TKO'] - ufc['B_win_by_KO/TKO']
ufc['win_by_Submission_dif'] = ufc['R_win_by_Submission'] - ufc['B_win_by_Submission']
ufc['win_by_TKO_Doctor_Stoppage_dif'] = ufc['R_win_by_TKO_Doctor_Stoppage'] - ufc['B_win_by_TKO_Doctor_Stoppage']
ufc['wins_dif'] = ufc['R_wins'] - ufc['B_wins']
ufc['Height_cms_dif'] = ufc['R_Height_cms'] - ufc['B_Height_cms']
ufc['Reach_cms_dif'] = ufc['R_Reach_cms'] - ufc['B_Reach_cms']
ufc['Weight_lbs_dif'] = ufc['R_Weight_lbs'] - ufc['B_Weight_lbs']
ufc['age_dif'] = ufc['R_age'] - ufc['B_age']
ufc['dec_odds_dif'] = ufc['r_dec_odds'] - ufc['b_dec_odds']
ufc['sub_odds_dif'] = ufc['r_sub_odds'] - ufc['b_sub_odds']
ufc['ko_odds_dif'] = ufc['r_ko_odds'] - ufc['b_ko_odds']

col2 = [
    'R_odds','B_odds','R_ev','B_ev','R_current_lose_streak','B_current_lose_streak',
    'R_current_win_streak', 'B_current_win_streak', 'R_draw', 'B_draw',
    'R_avg_SIG_STR_landed','B_avg_SIG_STR_landed','R_avg_SIG_STR_pct','B_avg_SIG_STR_pct',
    'R_avg_SUB_ATT', 'B_avg_SUB_ATT', 'R_avg_TD_landed','B_avg_TD_landed',
    'R_avg_TD_pct', 'B_avg_TD_pct', 'R_longest_win_streak', 'B_longest_win_streak',
    'R_losses', 'B_losses', 'R_total_rounds_fought', 'B_total_rounds_fought',
    'R_total_title_bouts', 'B_total_title_bouts',
    'R_win_by_Decision_Majority','B_win_by_Decision_Majority',
    'R_win_by_Decision_Split', 'B_win_by_Decision_Split',
    'R_win_by_Decision_Unanimous', 'B_win_by_Decision_Unanimous',
    'R_win_by_KO/TKO', 'B_win_by_KO/TKO', 'R_win_by_Submission', 'B_win_by_Submission',
    'R_win_by_TKO_Doctor_Stoppage', 'B_win_by_TKO_Doctor_Stoppage', 'R_wins','B_wins',
    'R_Height_cms', 'B_Height_cms', 'R_Reach_cms', 'B_Reach_cms', 'R_Weight_lbs', 'B_Weight_lbs',
    'r_dec_odds','b_dec_odds', 'r_sub_odds', 'b_sub_odds','r_ko_odds','b_ko_odds', 
    'R_age','B_age'
]
ufc.drop(col2, axis=1, inplace=True)

col3 = [
 'B_match_weightclass_rank',
 'R_match_weightclass_rank',
 "R_Women's Flyweight_rank",
 "R_Women's Featherweight_rank",
 "R_Women's Strawweight_rank",
 "R_Women's Bantamweight_rank",
 'R_Heavyweight_rank',
 'R_Light Heavyweight_rank',
 'R_Middleweight_rank',
 'R_Welterweight_rank',
 'R_Lightweight_rank',
 'R_Featherweight_rank',
 'R_Bantamweight_rank',
 'R_Flyweight_rank',
 'R_Pound-for-Pound_rank',
 "B_Women's Flyweight_rank",
 "B_Women's Featherweight_rank",
 "B_Women's Strawweight_rank",
 "B_Women's Bantamweight_rank",
 'B_Heavyweight_rank',
 'B_Light Heavyweight_rank',
 'B_Middleweight_rank',
 'B_Welterweight_rank',
 'B_Lightweight_rank',
 'B_Featherweight_rank',
 'B_Bantamweight_rank',
 'B_Flyweight_rank',
 'B_Pound-for-Pound_rank',
]
ufc.drop(col3, axis=1, inplace=True)

# Now we have a feat feature vector and a ys for the classes 1 = red won
ys = ufc['Winner'].copy()
feat = ufc.drop(columns=['Winner'])


# split data
#CHANGED: Using reference paper data here
X_train, X_test, y_train, y_test = train_test_split(feat, ys, test_size=0.3, random_state=1234)

# Convert pandas data to numpy arrays
X_train = X_train.to_numpy().astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)

# scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y tensors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Create model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

n_features = X_train.shape[1]
model = LogisticRegression(n_features)

#Loss and optimizer
#Changes: modifyed learin_rate from 0.01 to 0.1
learning_rate = 0.1
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
#change: Modified num of epochs from 500 to 4000
num_epochs = 3000

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()
    # updates
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()
    # Change to modify rate we print our loss at: every_print = ...
    every_print = 250
    if (epoch+1) % every_print == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])  # accuracy
    print(f'accuracy = {acc:.4f}')

    y_true = y_test.view(-1).numpy()
    y_pred = y_predicted_cls.view(-1).numpy()

    # Calculate precision
    precision = precision_score(y_true, y_pred)
    print(f'Precision: {precision:.4f}')

    ###################################################

    # chatgpt code: monitoring validation loss (trying to understand if we're potentially overfitting)
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset

# # Assuming you have X_train, X_test, y_train, y_test tensors ready
# # Create datasets for train and validation
# train_data = TensorDataset(X_train, y_train)
# val_data = TensorDataset(X_test, y_test)

# # Create DataLoaders
# batch_size = 32
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

# # # Initialize your model, loss criterion, and optimizer
# # model = LogisticRegression(X_train.shape[1])
# # criterion = torch.nn.BCELoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# def train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer):
#     for epoch in range(num_epochs):
#         model.train()
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             val_loss = 0
#             for X_val, y_val in val_loader:
#                 val_outputs = model(X_val)
#                 val_loss += criterion(val_outputs, y_val).item()

#         val_loss /= len(val_loader)
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# train_model(100, model, train_loader, val_loader, criterion, optimizer)

##################### NEURAL NETS FROM CHATGPT #########################

class SimpleNN(nn.Module): 
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(input_dim, 128)  # 128 nodes in the first hidden layer
        # Define the second hidden layer
        self.hidden2 = nn.Linear(128, 64)  # 64 nodes in the second hidden layer
        # Output layer
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # Forward pass through the first hidden layer
        x = F.relu(self.hidden1(x))
        # Forward pass through the second hidden iyer
        x = F.relu(self.hidden2(x))
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output(x))
        return x
    
class ImprovedNN(nn.Module): 
    def __init__(self, input_dim):
        super(ImprovedNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer
        self.hidden2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)  # Another Dropout layer
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        x = torch.sigmoid(self.output(x))
        return x


# Setup optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
# model = SimpleNN(input_dim=X_train.shape[1]) # Accuracy: 0.6549 accuracy
model = ImprovedNN(input_dim=X_train.shape[1]) # Accuracy: 0.6535

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

# Number of epochs
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Clear the gradients
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # Backward pass
    loss.backward()
    optimizer.step()  # Update the weights

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.round()  # Threshold the outputs to get binary results
    accuracy = (predictions.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f'Accuracy: {accuracy:.4f}')