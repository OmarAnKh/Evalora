import json
from pathlib import Path
from pydantic import ValidationError

from src.schemas.dataset import EvaluationSample


class Formatter:
    """
    Formats EvaluationSample records into
    Mistral instruction-tuning text samples.
    """

    def format(
        self,
        record: EvaluationSample,
    ) -> dict[str, str]:
        """Formats a single EvaluationSample record into a Mistral instruction-tuning text sample.
        Args:
            record (EvaluationSample): The evaluation sample record to format.
        Returns:
            dict[str, str]: A dictionary containing the formatted text for the evaluation sample.
        """
        rubric_text = self._format_rubric(record.rubric)
        prompt = f"""<s>[INST]
        You are an automated evaluation system.

        Task:
        {record.task}

        Reference Answer:
        {record.reference_answer}

        Student Answer:
        {record.answer}

        Rubric:
        {rubric_text}

        Evaluate the student answer and return:
        - score
        - reasoning

        [/INST]

        {{
        "score": {record.score},
        "reasoning": "{record.reasoning}"
        }}
        </s>"""

        return {"text": prompt}

    def format_batch(self, records: list[EvaluationSample]) -> list[dict[str, str]]:
        """Formats a batch of EvaluationSample records.
        Args:
            records (list[EvaluationSample]): A list of EvaluationSample records to format.
        Returns:
            list[dict[str, str]]: A list of dictionaries containing the formatted text for each record.
        """

        return [self.format(record) for record in records]

    def _format_rubric(
        self,
        rubric,
    ) -> str:
        """Formats the rubric criteria into a readable string format.
        Args:
            rubric (list[Criterion]): A list of Criterion objects representing the evaluation rubric.
        Returns:
            str: A formatted string representation of the rubric criteria.
        """
        return "\n".join(
            (f"- {item.criterion} " f"(weight: {item.weight}): " f"{item.description}")
            for item in rubric
        )

    def format_file(self, file_id: str) -> list[dict[str, str]]:
        """Formats evaluation sample records from a specified file.
        Args:
            file_id (str): The identifier of the file containing evaluation sample records to format.
        Returns:
            list[dict[str, str]]: A list of dictionaries containing the formatted text for each record in the file.
        """

        sub_path = "processed" if file_id.startswith("processed_") else "raw"
        normalized_file_id = (
            file_id if file_id.endswith(".jsonl") else f"{file_id}.jsonl"
        )

        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / sub_path / normalized_file_id

        if not path.exists():
            # Backward-compatible fallback for callers that pass an exact filename.
            fallback_path = repo_root / "data" / sub_path / file_id
            if fallback_path.exists():
                path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Input file not found: {file_id} (also tried: {normalized_file_id})"
                )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            records: list[EvaluationSample] = []

            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} in {file_id}: {exc}"
                    ) from exc

                try:
                    records.append(EvaluationSample.model_validate(payload))
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid record at line {line_number} in {file_id}: {exc}"
                    ) from exc

            return self.format_batch(records)
