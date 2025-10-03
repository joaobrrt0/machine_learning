#%%
import pandas as pd

url = "https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3BiTXFhSkxtalZZV3g0U0xWVGNXYjFWOTRFZ3xBQ3Jtc0tsVjFoSjZXSWU5ZVotb0J0cGtQbDFmWjFjY004UHBydzdPZFhueHdNU1lLdHEzTFh3dXBEa1k2SkhpZ3ZmeXdxMkZnNFoxY3dYSnMzVDVNOEpidUFHV3AxMnMyaUJpZGg0cHJzdTFtb2FfWHEtVFRvTQ&q=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2F1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw%2Fexport%3Fgid%3D283387421%26format%3Dcsv&v=ImWgtWmP61s"

df = pd.read_csv(url)
df.head()

# %%
