from pathlib import Path

import numpy as np
import pandas as pd
from qcore import nhm

from pygmt_helper import plots, plotting

# Config
nhm_ffp = Path("path to nhm file")
output_dir = Path("./")

# If true, then for each CS fault
# all realisation hypocentres are plotted
show_hypo = False

# Load the NHM
nhm_data = nhm.load_nhm(str(nhm_ffp))

# Load plotting data
map_data = plotting.NZMapData.load()

# Load the source information
cybershake_df = pd.read_csv(
    Path(__file__).parent / "resources" / "cybershake_v20p4_200.csv", index_col=0
)
cybershake_df["fault"] = np.stack(
    np.char.split(cybershake_df.index.values.astype(str), "_", maxsplit=1), axis=1
)[0, :]
cybershake_df["historic"] = False

small_df = pd.read_csv(
    Path(__file__).parent / "resources" / "val_small.csv", index_col=0
)
small_df["fault"] = small_df.index.values.astype(str)
small_df["historic"] = True

rupture_df = pd.concat((cybershake_df, small_df), axis=0)

# Generate the plot & save
fig = plots.faults_plot(
    rupture_df,
    [cur_fault for cur_name, cur_fault in nhm_data.items()],
    map_data=map_data,
    title="Faults",
    show_hypo=show_hypo,
)
fig.savefig(
    output_dir / f"fault_traces.png",
    dpi=900,
    anti_alias=True,
)