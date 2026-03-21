2000 - 2021 Stock Dataset Analysis

DATASET SOURCE: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

## Project Review:

### Date
- Standardized date column

### Missing Values
- Deleted 649 rows with missing values

### Duplicates
- Deleted all duplicates

### Logic test
- Got 1264 invalid rows
- Deleted it
- A significant number of records had invalid Open prices equal to zero
it has been treated as missing values and removed to ensure data integrity.


1. Как ведут себя цены?
2. Как распределены доходности?
3. Есть ли волатильность кластерами?
4. Есть ли зависимости (autocorrelation)?