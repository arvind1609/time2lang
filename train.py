import pandas as pd
import numpy as np
import torch
import joblib
import sys 
import os
import argparse
sys.path.append("../time2lang/models")

from tqdm import tqdm
from chronos import ChronosPipeline
from utils import save_model, dictionary_to_arrays_numpy, zscore_sample_wise
from resnet import ResNet1D, ResNet1Dv2
from llama import CustomLlamaForCausalLM
from projection import ProjectionModel, ProjectionModelRes
from torchinfo import summary
from torch import nn
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from transformers.models.llama import LlamaForCausalLM
from transformers import pipeline, AutoTokenizer
from itertools import chain

def train_step(resnet, chronos, llama, projection, dataloader, loss_fn, optimizer, device):

    """
    Trains the model for a single epoch

    Args:
        resnet (torch.nn.Module): The input encoder (f) which is a 1D-CNN style resnet
        chronos (Chronos Pipelie with torch models): A time series foundation model pipeline
        llama (transformers.models.llama.LlamaForCausalLM): An LLM that is modified to recieve input embeddings
        projection (torch.nn.Module): The output mapping layer or projection (g) which is a fully-connected NN
        dataloader (torch.utils.data.DataLoader): Training dataloader for batch processing
        loss_fn (torch.nn.CrossEntropyLoss): Cross-entropy loss for multi-class classification
        optimizer (torch.optim): Adam optimizer
        device (str): GPU device 
    """

    resnet.train()
    chronos.model.eval()
    llama.model.eval()
    projection.train()
    train_loss, train_acc = 0, 0 

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        chronos_embeddings = chronos.embed(X)[0].permute(0, 2, 1).to(device) # swap context and ts length dimensions: Output=(batch, channels, context_length)
        resnet_embeddings = resnet(chronos_embeddings).permute(0, 2, 1) # shape = (batch_size, length, feature_dims)
        pad_embeddings = torch.nn.functional.pad(resnet_embeddings, (0, 2048 - resnet_embeddings.shape[-1])) # shape = (batch_size, length, feature_dims)
        llama_embeddings = llama(inputs_embeds=pad_embeddings) # shape = (batch_size, length, feature_dims)
        embeddings = projection(llama_embeddings.mean(dim=1), chronos_embeddings.mean(dim=-1)) # embeddings_shape=(batch_size, n_classes or other) 

        y = y.to(device)
        loss = loss_fn(embeddings, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred = torch.argmax(torch.softmax(embeddings, dim=1), dim=1).cpu()
        train_acc += (y_pred == y.cpu()).sum().item()/len(embeddings)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(resnet, chronos, llama, projection, dataloader, loss_fn, device):

    """
    Tests the model for a single epoch

    Args:
        resnet (torch.nn.Module): The input encoder (f) which is a 1D-CNN style resnet
        chronos (Chronos Pipelie with torch models): A time series foundation model pipeline
        llama (transformers.models.llama.LlamaForCausalLM): An LLM that is modified to recieve input embeddings
        projection (torch.nn.Module): The output mapping layer or projection (g) which is a fully-connected NN
        dataloader (torch.utils.data.DataLoader): Validation dataloader for batch processing
        loss_fn (torch.nn.CrossEntropyLoss): Cross-entropy loss for multi-class classification
        device (str): GPU device 
    """
    resnet.eval()
    chronos.model.eval()
    llama.model.eval()
    projection.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            chronos_embeddings = chronos.embed(X)[0].permute(0, 2, 1).to(device) 
            resnet_embeddings = resnet(chronos_embeddings).permute(0, 2, 1)
            pad_embeddings = torch.nn.functional.pad(resnet_embeddings, (0, 2048 - resnet_embeddings.shape[-1])) 
            llama_embeddings = llama(inputs_embeds=pad_embeddings) 
            embeddings = projection(llama_embeddings.mean(dim=1), chronos_embeddings.mean(dim=-1)) 

            y = y.to(device)
            test_loss += loss_fn(embeddings, y).item()
            test_pred = embeddings.argmax(dim=1).cpu()
            test_acc += ((test_pred == y.cpu()).sum().item()/len(test_pred))

            del chronos_embeddings, resnet_embeddings, pad_embeddings
            torch.cuda.empty_cache()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(resnet, chronos, llama, projection, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, base_dir, model_name, device):
    """
    Main training function

    Args:
        resnet (torch.nn.Module): The input encoder (f) which is a 1D-CNN style resnet
        chronos (Chronos Pipelie with torch models): A time series foundation model pipeline
        llama (transformers.models.llama.LlamaForCausalLM): An LLM that is modified to recieve input embeddings
        projection (torch.nn.Module): The output mapping layer or projection (g) which is a fully-connected NN
        train_dataloader (torch.utils.data.DataLoader): Training dataloader for batch processing
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader for batch processing
        loss_fn (torch.nn.CrossEntropyLoss): Cross-entropy loss for multi-class classification
        epochs (int): Total training epochs
        base_dir (string): Path to save checkpoints
        model_name (string): Name of the model for saving 
        device (str): GPU device 
    """
    
    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(resnet=resnet, 
                                      chronos=chronos, 
                                      llama=llama,
                                      projection=projection,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn, 
                                      optimizer=optimizer,
                                      device=device)
        
        test_loss, test_acc = test_step(resnet=resnet, 
                              chronos=chronos, 
                              llama=llama,
                              projection=projection,
                              dataloader=val_dataloader,
                              loss_fn=loss_fn, 
                              device=device)
    
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        save_model(model=resnet, base_dir=base_dir, name=f"resnet_{model_name}_{str(epoch)}_{test_acc:2f}")
        save_model(model=projection, base_dir=base_dir, name=f"proj_{model_name}_{str(epoch)}_{test_acc:2f}")

    # Return the filled results at the end of the epochs
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True, help="GPU CUDA id")
    parser.add_argument("--batch_size", type=int, required=True, help="dataloader batch size")
    parser.add_argument("--epochs", type=int, required=True, help="No. of training epochs")
    parser.add_argument("--save_dir", type=str, required=True, help="save directory for models")
    parser.add_argument("--model_name", type=str, required=True, help="name of the model")
    args = parser.parse_args()
    
    ### Data ###
    train_dataset = joblib.load(f"data/train.p")
    val_dataset = joblib.load(f"data/val.p")
    test_dataset = joblib.load(f"data/test.p")
        
    enc = LabelEncoder() # Convert period values to labels
    train_labels = enc.fit_transform(train_dataset['labels'])
    val_labels = enc.transform(val_dataset['labels'])
    test_labels = enc.transform(test_dataset['labels'])

    n_classes = len(np.unique(train_labels))

    train_dataset = TensorDataset(torch.tensor(train_dataset['ts'], dtype=torch.float32), 
                              torch.tensor(train_labels, dtype=torch.long))

    val_dataset = TensorDataset(torch.tensor(val_dataset['ts'], dtype=torch.float32), 
                                  torch.tensor(val_labels, dtype=torch.long))
    
    test_dataset = TensorDataset(torch.tensor(test_dataset['ts'], dtype=torch.float32), 
                                  torch.tensor(test_labels, dtype=torch.long))
    batch_size = args.batch_size
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ### Models ##
    device = "cuda:" + args.gpu if torch.cuda.is_available() else "cpu"
    resnet = ResNet1Dv2(in_channels=768,
                     base_filters=32,
                     kernel_size=3,
                     stride=2,
                     n_block=6,
                     groups=1,
                     n_classes=999).to(device) #classes dont matter
    
    chronos = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    for param in chronos.model.parameters():
        param.requires_grad = False
    
    model_id = "meta-llama/Llama-3.2-1B"
    llama = CustomLlamaForCausalLM.from_pretrained(model_id).to(device)
    projection = ProjectionModelRes(n_classes=n_classes).to(device)
    
    for param in llama.model.parameters():
        param.requires_grad = False

    ### Training ###
    epochs = args.epochs
    lr = 5e-4
    base_dir = args.save_dir
    model_name = args.model_name
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=chain(resnet.parameters(), projection.parameters()), lr=lr)

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    results = train(resnet=resnet,
                   chronos=chronos,
                   llama=llama,
                   projection=projection,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   epochs=epochs,
                   base_dir=base_dir,
                   model_name=model_name,
                   device=device)
            
    

