import datetime
import itertools
import json
import os
import pathlib
import typing

from core import utils, xlsx

FULL_STATS_KEYS = ['NI', 'F_found', 'F_avg', 'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'I_avg', 'GR_early',
                   'GR_avg', 'GR_late', 'NI_GR_late', 'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
                   'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg', 's_min', 'NI_s_min', 's_max',
                   'NI_s_max', 's_avg']
FULL_TOTAL_STATS_KEYS = ['Suc', 'Min_NI', 'Max_NI', 'Avg_NI', 'Sigma_NI', 'Min_I_min', 'NI_I_min', 'Max_I_max',
                         'NI_I_max', 'Avg_I_min', 'Avg_I_max', 'Avg_I_avg', 'Sigma_I_max', 'Sigma_I_min',
                         'Sigma_I_avg', 'AvgGR_early', 'MinGR_early', 'MaxGR_early', 'AvgGR_late', 'MinGR_late',
                         'MaxGR_late', 'AvgGR_avg', 'MinGR_avg', 'MaxGR_avg', 'Min_RR_min', 'NI_RR_min',
                         'Max_RR_max', 'NI_RR_max', 'Avg_RR_min', 'Avg_RR_max', 'Avg_RR_avg', 'Sigma_RR_max',
                         'Sigma_RR_min', 'Sigma_RR_avg', 'Min_Teta_min', 'NI_Teta_min', 'Max_Teta_max',
                         'NI_Teta_max', 'Avg_Teta_min', 'Avg_Teta_max', 'Avg_Teta_avg', 'Sigma_Teta_max',
                         'Sigma_Teta_min', 'Sigma_Teta_avg', 'Min_s_min', 'NI_s_min', 'Max_s_max', 'NI_s_max',
                         'Avg_s_min', 'Avg_s_max', 'Avg_s_avg']

NOISE_STATS_KEYS = ['NI', 'ConvTo']
NOISE_TOTAL_STATS_KEYS = ["Suc", "Num0", "Num1", "Min_NI", "Max_NI", "Avg_NI", "Sigma_NI"]


class ReportBuilder:
    def __init__(self, report_dir: str, reports: typing.Union[list, None] = None):
        self._reports: list = reports
        self._report_dir: pathlib.Path = pathlib.Path(f"./{report_dir}/data")
        self._writing_dir: pathlib.Path = pathlib.Path(f"./{report_dir}/xlsx")

        self._writing_dir.mkdir(parents=True, exist_ok=True)

        if not self._reports:
            self._reports = [
                str(f) for f in os.listdir(self._report_dir)
                if os.path.isfile(os.path.join(self._report_dir, f))
            ]

        self._writer = xlsx.XLSX(
            str(self._writing_dir / f"report_{datetime.datetime.now().isoformat()}.xlsx"),
            "EP Report Sheet"
        )

        self._default_skip_rows_after_report = 7

    def _read_report(self, report_name: str):
        with open(self._report_dir / report_name, 'rb') as f:
            return json.load(f)

    def _write_report(self, report_data: dict):
        n = report_data["n"]
        epochs = report_data["epochs"]
        max_iteration = report_data["max_iteration"]
        selection_fns = report_data["selection_fns"]
        length = report_data["length"]
        fitness_fn = report_data["fitness_fn"]
        fitness_fn_values = report_data["fitness_fn_values"]
        stats_mode = report_data["stats_mode"]
        data = report_data["data"]

        if "total_data" not in report_data or not report_data["total_data"]:
            total_data = utils.aggregate_runs_data(data, stats_mode)
        else:
            total_data = report_data["total_data"]

        stats_keys = NOISE_STATS_KEYS if stats_mode == "noise" else FULL_STATS_KEYS
        total_stats_keys = NOISE_TOTAL_STATS_KEYS if stats_mode == "noise" else FULL_TOTAL_STATS_KEYS
        selection_fns = list(itertools.product(selection_fns["beta"], selection_fns["modified"]))

        self._writer.text(
            f"{fitness_fn}, n={n}, length={length}, max_iteration={max_iteration}, fitness_params={fitness_fn_values}",
            style="bold_bg"
        )

        self._writer.col(
            ['Epochs', 'Selection \\ Criteria',
             *[f"linear{' modified' if modified else ''}, beta={beta}"
               for beta, modified in selection_fns]],
            style="bold_bg"
        )

        for epoch in range(epochs):
            epoch_to_write = True
            for stat in stats_keys:
                lst = [f"epoch {epoch + 1}" if epoch_to_write else '', stat]

                for beta, modified in selection_fns:
                    lst.append(data[epoch][f"{beta}${modified}"][stat])

                self._writer.col(lst, style="bold_bg" if epoch_to_write else "normal")

                epoch_to_write = False

        # WRITE TOTAL DATA
        write_title = True
        for stat in total_stats_keys:
            lst = [f"Total stats" if write_title else '', stat]

            for beta, modified in selection_fns:
                lst.append(total_data[f"{beta}${modified}"][stat])

            self._writer.col(lst, style="bold_bg" if write_title else "normal")

            write_title = False

        self._writer.set_pos(col=1)
        self._writer.skip(row=len(selection_fns) + self._default_skip_rows_after_report)

    def generate(self):
        for report in self._reports:
            self._write_report(self._read_report(report))

        self._writer.save()


if __name__ == "__main__":
    ReportBuilder("../reports").generate()
