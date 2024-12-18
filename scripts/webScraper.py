import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URLs
BIOPROJECT_URL = "https://www.ncbi.nlm.nih.gov/sra?LinkName=bioproject_sra_all&from_uid=762199"
SRX_BASE_URL = "https://www.ncbi.nlm.nih.gov/sra/"

# Function to fetch SRX IDs from the BioProject page
def get_srx_ids(bioproject_url):
    response = requests.get(bioproject_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract SRX IDs (assuming they are linked on the page)
    srx_links = soup.find_all("a", href=True)
    srx_ids = [link.text for link in srx_links if "SRX" in link.text]
    return srx_ids

# Function to scrape data from each SRX page
def get_srx_data(srx_id):
    url = SRX_BASE_URL + srx_id
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Example: Extract a specific text field (modify this based on the page structure)
    title = soup.find("h1").text.strip() if soup.find("h1") else "N/A"
    description = soup.find("meta", attrs={"name": "description"})
    description_text = description["content"] if description else "N/A"
    
    return {"SRX": srx_id, "Title": title, "Description": description_text}

# Main function to gather data and save it to a CSV
def scrape_bioproject_to_csv(bioproject_url, output_file):
    srx_ids = get_srx_ids(bioproject_url)
    data = []

    for srx_id in srx_ids:
        print(f"Processing {srx_id}...")
        srx_data = get_srx_data(srx_id)
        data.append(srx_data)

    # Write data to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["SRX", "Title", "Description"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {output_file}")

# Run the script
if __name__ == "__main__":
    output_csv = "bioproject_srx_data.csv"
    scrape_bioproject_to_csv(BIOPROJECT_URL, output_csv)
