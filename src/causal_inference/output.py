"""推定結果の図・表の保存。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from util.paths import ensure_parent_dir, outputs_dir


def default_run_id() -> str:
    """実行ごとの出力フォルダ名（日時。例: `20261115_153000`）を返す。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class OutputSaver:
    """図・表を実行ごとのフォルダに保存する。

    図は `outputs/figures/<分析名>/<実行ID>/`、表は `outputs/tables/<分析名>/<実行ID>/` に保存する。

    実行 ID ごとにフォルダを分けるので、前の実行の出力は上書きしない。

    Attributes:
        figure_dir: 図の保存先。
        table_dir: 表の保存先。
        paths: 保存したファイルのパス（保存した順）。
    """

    figure_dir: Path
    table_dir: Path
    paths: list[Path] = field(default_factory=list)

    @classmethod
    def create(
        cls, analysis_name: str, run_id: str | None = None, root: Path | None = None
    ) -> OutputSaver:
        """保存先を決めて作る。

        Args:
            analysis_name: 分析名（フォルダ名）。
            run_id: 実行 ID（None なら日時）。
            root: 出力のルート（None なら `outputs/`）。

        Returns:
            保存先を設定した `OutputSaver`。
        """
        base = root if root is not None else outputs_dir()
        run_id = run_id or default_run_id()
        return cls(
            figure_dir=base / "figures" / analysis_name / run_id,
            table_dir=base / "tables" / analysis_name / run_id,
        )

    def figure(self, fig: plt.Figure, filename: str, *, close: bool = True, dpi: int = 150) -> Path:
        """図を PNG で保存する。

        Args:
            fig: 保存する図。
            filename: ファイル名（拡張子なしなら `.png` を付ける）。
            close: 保存後に図を閉じるか（ノートブックでは表示してから閉じる）。
            dpi: 解像度。

        Returns:
            保存したパス。
        """
        path = ensure_parent_dir(self.figure_dir / _with_suffix(filename, ".png"))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig)
        self.paths.append(path)
        return path

    def table(self, frame: pd.DataFrame, filename: str, *, index: bool = False) -> Path:
        """表を CSV（UTF-8 BOM 付き。Excel で文字化けしない）で保存する。

        Args:
            frame: 保存する表。
            filename: ファイル名（拡張子なしなら `.csv` を付ける）。
            index: index も保存するか。

        Returns:
            保存したパス。
        """
        path = ensure_parent_dir(self.table_dir / _with_suffix(filename, ".csv"))
        frame.to_csv(path, index=index, encoding="utf-8-sig")
        self.paths.append(path)
        return path


def _with_suffix(filename: str, suffix: str) -> str:
    return filename if Path(filename).suffix else f"{filename}{suffix}"
