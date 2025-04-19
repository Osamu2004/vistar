# Define weight URLs for different models, organized by dataset
weight_urls = {
    'imagenet1k_v1': {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
        'resnet50': "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        'resnet101':"https://download.pytorch.org/models/resnet101-63fe2227.pth",
        'resnext50_32x4d':"https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        'resnext101_32x8d':"https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"



    },
    'imagenet1k_v2':{
        'resnet50': "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        'resnet101':"https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        'resnext50_32x4d':"https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        'resnext101_32x8d':"https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"
    }
}

# Function to list all available datasets
def list_datasets():
    print(list(weight_urls.keys())) 
# Function to list all available models across all datasets
def list_models():
    models = []
    for dataset, model_dict in weight_urls.items():
        models.extend(model_dict.keys())
    print(models)

# Function to list all models for a specific dataset
def list_models_for_dataset(dataset_name):
    if dataset_name in weight_urls:
        print(list(weight_urls[dataset_name].keys()))
    else:
        print(f"Dataset {dataset_name} not found.")
    
def list_datasets_for_model(model_name):
    datasets = []
    for dataset, model_dict in weight_urls.items():
        if model_name in model_dict:
            datasets.append(dataset)
    print(datasets if datasets else f"Model {model_name} not found in any dataset.")


def get_url(dataset_name, model_name):
    # Check if the dataset exists in the weight_urls dictionary
    if dataset_name in weight_urls:
        dataset = weight_urls[dataset_name]
        
        # Check if the model exists in the specified dataset
        if model_name in dataset:
            return dataset[model_name]
        else:
            return f"Model '{model_name}' not found in dataset '{dataset_name}'."
    else:
        return f"Dataset '{dataset_name}' not found."

