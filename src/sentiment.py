#========================================================#
#=============> Calculate Sentiment Scores <=============#
#========================================================#

#=====> Import modules
# Data analysis
import os
import pandas as pd
from tqdm import tqdm # For telling me how long there is left for my code to run (Progress bar)
import numpy as np

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer() # Initializing VADER
# sentiment with spacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob') # Add to spacy pipeline

# visualisations
import matplotlib.pyplot as plt

#=====> Define functions
# > Load Data
def load_data(filename, subset):
    # Print info 
    print("[INFO] Loading data...")
    
    # Get the filepath
    filepath = os.path.join("in", filename)
    # Reading the filepath 
    data = pd.read_csv(filepath)
    
    # Subset dataset
    sub_data = data[data["label"] == subset]
    # reset index
    sub_data = sub_data.reset_index()
    
    return sub_data

# > Compute VADER sentment scores 
def vader_sentiment(data, subset):
    # Print info
    print("[INFO] Computing VADER sentiment scores...")
    
    # Define empthy list
    sentiment_data = []

    # for-loop
    for headline, ID in tqdm(zip(data["title"], data["Unnamed: 0"])):
        # Get sentiment scores
        sentiment = analyzer.polarity_scores(headline)
        
        # Create a doc
        doc = nlp(headline)
        # Get GPEs
        for entity in doc.ents:
            if entity.label_ == "GPE":
                # Append to list
                sentiment_data.append((ID, sentiment["neg"], sentiment["neu"], sentiment["pos"], sentiment["compound"], entity.text))
    
    # Create dataframe 
    sentiment_df = pd.DataFrame(sentiment_data, columns = ["ID", "neg", "neu", "pos", "compound", "GPE"])
    
    # Save dataframe 
    outpath = os.path.join("output", f"{subset}_vader_sentiment.csv")
    sentiment_df.to_csv(outpath, index=False)
    
    return sentiment_df

# > Compute spacyTextBlob sentiment scores
def blob_sentiment(data, subset):
    # Print info 
    print("[INFO] Compiting spacyTextBlob sentiment scores...")
    
    # Define empthy list 
    sentiment_data = []

    # for-loop
    for doc, ID in tqdm(zip(nlp.pipe(data["title"], batch_size=500), data["Unnamed: 0"])):
        # Get GPEs
        for entity in doc.ents:
            if entity.label_ == "GPE":
                # Append sentiment scores to list 
                sentiment_data.append((ID, doc._.blob.polarity, doc._.blob.subjectivity, entity.text))
                
    # Create dataframe 
    sentiment_df = pd.DataFrame(sentiment_data, columns = ["ID", "polarity", "subjectivity", "GPE"])
    
    # Save dataframe 
    outpath = os.path.join("output", f"{subset}_blob_sentiment.csv")
    sentiment_df.to_csv(outpath, index=False)
    
    return sentiment_df

# > Plot occurances of the 20 most common GPEs
def freq_plot(data, subset):
    # Finding the 20 most common entities and their frequency in the dataset
    freq = data["GPE"].value_counts().values.tolist()[0:20]
    gpes = data["GPE"].value_counts().index.tolist()[0:20]
    
    # > Histogram
    # Create figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    # Label figure
    plt.title(f'{subset} News: Most common GPEs')
    plt.xlabel('GPEs', fontsize = 12)
    plt.ylabel('Occurances', fontsize = 12)
    plt.xticks(rotation = 60, fontsize = 9)

    # Draw bars
    ax.bar(gpes, freq)

    # Save figure
    plt.savefig(os.path.join("output", f"{subset}_freq_plot.png"), bbox_inches = "tight")
    
# > Plot VADER scores
def vader_plot(data, subset):
    # Get 20 most common GPEs
    gpes = data["GPE"].value_counts().index.tolist()[0:20]
    # Get mean values
    mean_df = data.groupby(["GPE"]).mean().reset_index()
    # Subset for most common GPEs
    sum_df = mean_df[mean_df.GPE.isin(gpes)]
    
    # Define static values
    r = np.arange(len(sum_df["GPE"]))
    width = 0.25

    # Create figure
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)

    # Add plot title
    plt.title(f"{subset} News: VADER scores", fontsize = 15)

    # Draw bars
    ax.bar(r, sum_df["neg"], width, label = "Negative")
    ax.bar(r + width, sum_df["neu"], width, label = "Neutral")
    ax.bar(r + (width*2), sum_df["pos"], width, label = "Positive")

    # Label plot
    plt.xticks(r + width/2, sum_df["GPE"], rotation = 45)
    plt.xlabel('GPEs', fontsize = 12)
    plt.ylabel('Score', fontsize = 12)
    plt.legend()

    # Save figure
    plt.savefig(os.path.join("output", f"{subset}_vader_plot.png"), bbox_inches = "tight")
    
# Plot spacyTextBlob scores
def blob_plot(data, subset):
    # Get 20 most common GPEs
    gpes = data["GPE"].value_counts().index.tolist()[0:20]
    # Get mean values
    mean_df = data.groupby(["GPE"]).mean().reset_index()
    # Subset for most common GPEs
    sum_df = mean_df[mean_df.GPE.isin(gpes)]
    
    # Define static values
    r = np.arange(len(sum_df["GPE"]))
    width = 1

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(15, 10)

    # Draw subplot 1
    ax1.set_title(f"{subset} News: spacyTextBlob scores", fontsize = 15)
    ax1.bar(sum_df["GPE"], sum_df["polarity"], width, label = "Polarity")
    ax1.tick_params(axis='x', rotation=60)
    ax1.set_ylabel('Score', fontsize = 12)

    # Draw subplot 2
    ax2.bar(sum_df["GPE"], sum_df["subjectivity"], width, label = "Subjectivity")
    ax2.tick_params(axis='x', rotation=60)
    ax2.set_ylabel('Score', fontsize = 12)

    # Label x-axis
    plt.xlabel('GPEs', fontsize = 12)
    # Adjust space between plots
    plt.subplots_adjust(hspace=0.4)

    # Save figure
    plt.savefig(os.path.join("output", f"{subset}blob_plot.png"), bbox_inches = "tight")

#=====> Define main()
def main():
    # Define list of types
    news_list = ["FAKE", "REAL"]
    
    # Iterate over types of news
    for news_type in news_list:
        # Print info 
        print(f"{news_type} News:")
    
        # Load data 
        sub_data = load_data("fake_or_real_news.csv", news_type)
        # Get VADER sentiment scores 
        vader_df = vader_sentiment(sub_data, news_type)
        # Get spacyTextBlob sentiment scores
        blob_df = blob_sentiment(sub_data, news_type)
    
        # Print info
        print("[INFO] Plotting...")
        
        # Create a frequency plot using one of the sentiment dataframes 
        # It shouldn't matter which one - the GPEs should be the same in both
        freq_plot(vader_df, news_type)
        # Plot VADER scores
        vader_plot(vader_df, news_type)
        # Plot spacyTextBlob scores
        blob_plot(blob_df, news_type)
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()
