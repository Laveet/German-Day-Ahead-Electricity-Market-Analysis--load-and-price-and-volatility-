import pandas as pd
from pathlib import Path
#1 accessing data directory folder
project_root = Path(__name__).resolve().parent.parent
raw_data_path= project_root / "data"/"raw"
clean_data_path=project_root / "data"/"clean"
clean_data_path.mkdir(parents=True, exist_ok=True)
price_files=list(raw_data_path.glob("Energy_price_*.csv")) ##getting all files saved in raw data folder
price_list= []
for file in price_files: #loading all files saved in data folders into a list and then concating all files to get one data frame 
    print(f"{file} is loaded")
    df=pd.read_csv(file)
    price_list.append(df)
price_df=pd.concat(price_list,ignore_index=True)
price_df=price_df.rename(columns={price_df.columns[0]:"timestamps","Day-ahead Price (EUR/MWh)":"Price"}) # renaming the columns
price_df=price_df[price_df['Sequence']=="Sequence Sequence 1"] ## Only want sequence 1 price 
price_df=price_df[['timestamps','Price']]
price_df['timestamps'] = pd.to_datetime(
    price_df['timestamps']
        .str.split(' - ')  # split start and end
        .str[0]            # take start
        .str.extract(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')[0],  # extract only datetime
    format='%d/%m/%Y %H:%M:%S')
price_df.sort_values('timestamps',inplace=True)
price_df.drop_duplicates(subset='timestamps')
price_df.set_index('timestamps',inplace=True)

output_file = clean_data_path / "prices_clean.csv"
price_df.to_csv(output_file)

print("Clean price data saved to:", output_file)
print("Final shape:", price_df.shape)
