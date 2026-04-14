from datasets import load_dataset
import pandas as pd

def get_imdb_data():
    """
    Loads the IMDb movie reviews dataset from Hugging Face.
    This fulfills our 'Gathering and Curating' requirement.
    """
    print("Downloading IMDb dataset from Hugging face...")

    #Load the dataset 
    dataset = load_dataset("imdb")

    #Convert to Pandas for easy preprocessing 
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    #Save a small subset to raw data folder to keep it lightweight for now
    train_df.sample(5000).to_csv("data/raw/imdb_train_subset.csv", index=False)

    print(f"Loaded {len(train_df)} training and {len(test_df)} test rows.")
    return train_df,test_df

if __name__ == '__main__':
    train_data, test_data = get_imdb_data()
    print(train_data.head())
    print(test_data.head())