import wntr
import pandas as pd
import numpy as np


class WaterNetworkAnalysis:
    def __init__(self, wn, leakage_min=19.94, leakage_max=199.4, leakage_levels=31):
        self.wn = wn
        self.leakage = list(np.linspace(leakage_min, leakage_max, leakage_levels))

    def analyze_and_store(self):
        # Prepare sorted pipe and node ID lists for consistent column ordering
        pipe_ids = sorted(self.wn.pipe_name_list, key=lambda x: int(x))
        node_ids = sorted(self.wn.junction_name_list, key=lambda x: int(x))

        wide_results = []
        total = len(node_ids) * len(self.leakage)

        for node_id_str in node_ids:
            node = self.wn.get_node(node_id_str)
            original_demand = node.demand_timeseries_list[0].base_value

            for i, leak in enumerate(self.leakage):
                current = (node_ids.index(node_id_str) * len(self.leakage)) + (i + 1)
                print(f"[{current}/{total}] Node: {node_id_str} | Level: {i+1} ({leak:.3f} L/s)")

                try:
                    # Apply leakage: convert L/s to m³/s
                    node.demand_timeseries_list[0].base_value = original_demand + leak / 1000

                    # Run simulation
                    sim = wntr.sim.EpanetSimulator(self.wn)
                    results = sim.run_sim(version=2.0)

                    # Record pipe flows at first time step (m³/s → L/s)
                    row = {
                        f"PIPE {pipe_id}": results.link["flowrate"][pipe_id].iloc[0] * 1000
                        for pipe_id in pipe_ids
                    }
                    row["NODE"] = node_id_str
                    row["LEAKAGE LEVEL"] = i + 1
                    wide_results.append(row)

                except Exception as e:
                    print(f"  ⚠️ Error at Node {node_id_str}, Level {i+1}: {e}")
                    row = {f"PIPE {pipe_id}": "Error" for pipe_id in pipe_ids}
                    row["NODE"] = node_id_str
                    row["LEAKAGE LEVEL"] = i + 1
                    wide_results.append(row)

                finally:
                    # Always reset demand, even if an error occurred
                    node.demand_timeseries_list[0].base_value = original_demand

        # Build DataFrame with consistent column order
        df = pd.DataFrame(wide_results)
        pipe_cols = [f"PIPE {pid}" for pid in pipe_ids]
        df = df[pipe_cols + ["NODE", "LEAKAGE LEVEL"]]

        output_file = "pipe_flows_AS_31.xlsx"
        df.to_excel(output_file, index=False)
        print(f"✅ Flow data saved to '{output_file}'.")


if __name__ == "__main__":
    wn = wntr.network.WaterNetworkModel("ASnet2.inp")
    analysis = WaterNetworkAnalysis(wn)
    analysis.analyze_and_store()
