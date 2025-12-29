import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Local imports
from util import DSPCA, accuracy

warnings.filterwarnings("ignore")

# Configuration for Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
def create_dataset(test_index=[8, 12, 14], data_path='./TE_mat_data/', attr_path='./attribute_matrix.xlsx'):
    print(f"Loading data from {data_path}...")

    # Ensure paths exist
    if not os.path.exists(data_path) or not os.path.exists(attr_path):
        raise FileNotFoundError("Data files not found. Please check 'TE_mat_data' folder and 'attribute_matrix.xlsx'.")

    # Load MATLAB files
    data_list = []
    for i in range(1, 16):
        fname = f'd{i:02d}.mat'
        mat = loadmat(os.path.join(data_path, fname))['data']
        data_list.append(mat)

    # Load Attribute Matrix
    attribute_matrix_df = pd.read_excel(attr_path, index_col='no')
    attribute_matrix = attribute_matrix_df.values

    # Determine indices
    all_indices = set(np.arange(15))
    train_index = list(all_indices - set(test_index))
    test_index.sort()
    train_index.sort()

    print(f"Test classes (indices): {test_index}")
    print(f"Train classes (indices): {train_index}")

    # Helper to construct arrays
    def build_set(indices, repeat_count=480):
        labels = []
        attr_labels = []
        data_arr = []
        for idx in indices:
            labels += [idx] * repeat_count
            attr_labels += [attribute_matrix[idx, :]] * repeat_count
            data_arr.append(data_list[idx])

        return (
            np.column_stack(data_arr).T,  # Data
            np.row_stack(labels),  # Labels
            np.row_stack(attr_labels)  # Attribute Labels
        )

    traindata, trainlabel, train_attributelabel = build_set(train_index)
    testdata, testlabel, test_attributelabel = build_set(test_index)

    return (traindata, trainlabel, train_attributelabel,
            testdata, testlabel, test_attributelabel,
            attribute_matrix_df.iloc[test_index, :],
            attribute_matrix_df.iloc[train_index, :],
            attribute_matrix_df)


def feature_extraction(traindata, testdata, train_attributelabel, test_attributelabel, n_components=20):
    """
    Extracts features using Dual Supervised PCA (DSPCA) for each attribute dimension.
    """
    trainfeatures = []
    testfeatures = []
    for i in range(train_attributelabel.shape[1]):
        spca = DSPCA(n_components)
        spca.fit(traindata, train_attributelabel[:, i])
        trainfeatures.append(spca.transform(traindata))
        testfeatures.append(spca.transform(testdata))
    return np.column_stack(trainfeatures), np.column_stack(testfeatures)


def remap_labels(labels):
    """Remaps non-contiguous class labels to 0..N range."""
    labels = np.squeeze(labels)
    unique_labels = np.unique(labels)
    old_to_new_mapping = {label: i for i, label in enumerate(unique_labels)}
    new_labels = np.array([old_to_new_mapping[label] for label in labels])
    return new_labels


# ==========================================
# 2. Classifier Logic (Attribute-Based)
# ==========================================
def predict_with_attributes(model_name, traindata, trainlabel, train_attributelabel, testdata, testlabel,
                            test_attributelabel, attribute_matrix):
    model_dict = {
        'SVC_linear': SVC(kernel='linear', probability=True),
        'lr': LogisticRegression(),
        'SVC_rbf': SVC(kernel='rbf', probability=True),
        'rf': RandomForestClassifier(n_estimators=10),
        'Ridge': Ridge(alpha=1),
        'NB': GaussianNB(),
        'Lasso': Lasso(alpha=0.1)
    }

    res_list = []
    # Train a binary classifier for each attribute column
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model_name]

        # Check if attribute has variance (not all zeros)
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(traindata, train_attributelabel[:, i])

            # Get Probabilities
            if hasattr(clf, 'predict_proba') and model_name in ['SVC_linear', 'lr', 'SVC_rbf', 'rf', 'NB']:
                res = clf.predict_proba(testdata)
                # Ensure shape is (N, 2)
                if res.shape[1] == 1:
                    res = np.column_stack([1 - res, res])
                elif res.shape[1] > 2:
                    pos_prob = res[:, 1:].sum(axis=1, keepdims=True)
                    neg_prob = res[:, 0:1]
                    res = np.column_stack([neg_prob, pos_prob])
            else:
                # Regressors (Ridge/Lasso) - Convert output to sigmoid probability
                res_pred = clf.predict(testdata)
                res_prob = 1 / (1 + np.exp(-res_pred))
                res_prob = np.clip(res_prob, 1e-7, 1 - 1e-7)
                res = np.column_stack([1 - res_prob, res_prob])
        else:
            # Default if attribute is always 0
            res = np.ones((testdata.shape[0], 2))
            res[:, 1] *= 0.001
            res[:, 0] *= 0.999

        if res.ndim == 1: res = res.reshape(-1, 1)
        if res.shape[1] == 1: res = np.column_stack([1 - res, res])

        res_list.append(res)

    # Concatenate all attribute probabilities
    # Shape: (n_samples, n_attributes * 2)
    test_pre_attribute = np.concatenate(res_list, axis=1)

    # Map attribute probabilities back to class labels
    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        # Initialize probability for each class as 1.0
        class_probs = [1.0] * attribute_matrix.shape[0]
        pre_res = test_pre_attribute[i, :]

        for j in range(attribute_matrix.shape[0]):  # For each class
            for k in range(attribute_matrix.shape[1]):  # For each attribute
                idx_0 = k * 2
                idx_1 = k * 2 + 1

                # Multiply probability based on whether class j has attribute k
                if attribute_matrix.iloc[j, k] == 0:
                    class_probs[j] *= pre_res[idx_0]
                else:
                    class_probs[j] *= pre_res[idx_1]

        # Select class with highest probability
        loc = np.array(class_probs).argmax()
        label_lis.append(attribute_matrix.index[loc] - 1)  # Adjust index to match label

    label_lis = np.array(label_lis).reshape(-1, 1)

    acc = accuracy_score(np.asarray(label_lis), np.asarray(testlabel))
    print(f"Model: {model_name} | Accuracy: {acc:.4f}")
    return label_lis, testlabel


# ==========================================
# 3. Neural Network Modules
# ==========================================
class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


class MLP(nn.Module):
    def __init__(self, latent_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.do = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.rl = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, 20)
        self.bn2 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.rl(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class CustomDataset(Dataset):
    def __init__(self, data, label, glabel):
        self.data = torch.tensor(data).type(torch.FloatTensor).to(device)
        self.label = torch.tensor(label).type(torch.FloatTensor).to(device)
        self.glabel = torch.tensor(glabel).type(torch.FloatTensor).to(device)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.glabel[index]

    def __len__(self):
        return len(self.data)


# ==========================================
# 4. Main Execution
# ==========================================
def run():
    # Hyperparameters
    ALPHA = 0.05  # Reconstruction Loss weight
    BETA = 0.5  # Center Loss weight
    GAMMA = 0.5  # NCE Loss weight

    # Dataset Selection
    # data_list = [1, 6, 14]  # Dataset A
    data_list = [4, 7, 10]  # Dataset B
    # data_list = [8, 11, 12] # Dataset C
    # data_list = [2, 3, 5]   # Dataset D

    BATCH_SIZE = 64
    HIDDEN_DIM = 256
    LATENT_DIM = 100
    EPOCHS = 100
    LR = 1e-2

    print(f"Running on: {device}")

    # 1. Prepare Data
    traindata, trainlabel, train_attributelabel, testdata, testlabel, \
    test_attributelabel, attribute_matrix, train_attribute_matrix, all_attribute_matrix = create_dataset(data_list)

    # 2. Extract DSPCA Features
    traindata, testdata = feature_extraction(traindata, testdata, train_attributelabel, test_attributelabel)

    # 3. Create Loader
    trainlabel_remapped = remap_labels(trainlabel)
    dataset = CustomDataset(traindata, train_attributelabel, trainlabel_remapped)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize Models
    # Input dim is 400 (20 attributes * 20 dim DSPCA)
    model = AE(input_dim=400, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    mlp = MLP(latent_dim=LATENT_DIM).to(device)
    centerloss = CenterLoss(num_classes=12, feat_dim=100).to(device)  # num_classes = 15 total - 3 test = 12

    optimizer = optim.RMSprop(list(model.parameters()) + list(mlp.parameters()) + list(centerloss.parameters()), lr=LR)

    # 5. Training Loop
    max_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        mlp.train()
        running_loss = 0.0

        for inputs, labels, glabel in dataloader:
            # Forward
            reconstruction, z = model(inputs)
            output = mlp(z)

            # Loss Calculation
            # 1. Reconstruction Loss (Cosine Similarity based)
            recon_sim = F.cosine_similarity(inputs, reconstruction)
            loss_recon = 1 - recon_sim.mean()

            # 2. NCE (Noise Contrastive Estimation) Loss - Compatibility
            p_compatibility = torch.sum(output * labels, 1, keepdim=True)
            train_attr_T = torch.tensor(train_attribute_matrix.values.T).type(torch.FloatTensor).to(device)
            n_compatibility = output @ train_attr_T

            loss_nce = -p_compatibility + torch.log(torch.sum(n_compatibility.exp(), 1, keepdim=True))
            loss_nce = loss_nce.mean()

            # 3. Center Loss
            loss_center = centerloss(z, glabel.unsqueeze(1))

            # Total Loss
            loss = (ALPHA * loss_recon) + (BETA * loss_center) + (GAMMA * loss_nce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluation
        model.eval()
        mlp.eval()
        with torch.no_grad():
            t_traindata = torch.tensor(traindata).type(torch.FloatTensor).to(device)
            t_testdata = torch.tensor(testdata).type(torch.FloatTensor).to(device)

            train_recon, train_z = model(t_traindata)
            test_recon, test_z = model(t_testdata)

            train_fea = mlp(train_z)
            test_fea = mlp(test_z)

            # Concatenate features: [Original, Original, Reconstruction, MLP_Features]
            # Note: The original code concatenated 'traindata_' (latent Z) but labeled it oddly.
            # Based on the concatenation logic: input + input + recon + mlp_out
            train_final = torch.cat([train_z, t_traindata, train_recon, train_fea], dim=1).cpu().numpy()
            test_final = torch.cat([test_z, t_testdata, test_recon, test_fea], dim=1).cpu().numpy()

            # Prediction using Gaussian Naive Bayes
            label_lis, _ = predict_with_attributes(
                'NB', train_final, trainlabel, train_attributelabel,
                test_final, testlabel, test_attributelabel, attribute_matrix
            )

            acc = accuracy_score(np.asarray(label_lis), np.asarray(testlabel))

            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), 'AE_best.pth')
                torch.save(mlp.state_dict(), 'MLP_best.pth')
                print(f"--> New Best Accuracy: {max_acc:.4f} (Saved Models)")

        print(
            f'Epoch: {epoch + 1}/{EPOCHS} | Loss: {running_loss:.4f} | Rec: {loss_recon:.4f} | NCE: {loss_nce:.4f} | Center: {loss_center:.4f}')

    print(f"Training Finished. Best Accuracy: {max_acc:.4f}")


if __name__ == '__main__':
    run()