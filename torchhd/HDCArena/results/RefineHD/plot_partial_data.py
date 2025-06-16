import pandas as pd

file = "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/RefineHD/online_iter_robust_partial_arena"

df = pd.read_csv(file)

latex = 0
pand = 1
df = df[df["robustness_failed_dimensions"] == 0]
df = df[df["dimensions"] == 10000]

mean_of_encoding = (
    df.sort_values(["partial_data"], ascending=False)
    .groupby(["partial_data", "name"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
)
mean_of_encoding = (
    mean_of_encoding.groupby(["partial_data"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
    .T
)
var_of_encoding = (
    df.sort_values(["partial_data"], ascending=False)
    .groupby(["partial_data", "name"])["accuracy"]
    .std()
    .round(3)
    .reset_index()
)
var_of_encoding = (
    var_of_encoding.groupby(["partial_data"])["accuracy"]
    .mean()
    .round(3)
    .reset_index()
    .T
)

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)

if latex:
    latex_table = mean_of_encoding.to_latex(
        index=False, caption="Encodings accuracy mean"
    )
    print(latex_table)
    # pd.options.display.float_format = "{:.2e}".format
    latex_table = var_of_encoding.to_latex(
        index=False, caption="Encodings accuracy variance"
    )
    print(latex_table)
    pd.options.display.float_format = None

    def join_with_commas(row):
        return ", ".join(str(value) for value in row)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # Apply the function to each row
    joined_values = mean_of_encoding.apply(join_with_commas, axis=1)

    # Print the joined values
    print(joined_values["partial_data"])
    print(joined_values["accuracy"])
