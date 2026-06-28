// ── Types ──

export interface PipelineResponse {
  upload_id: string;
  result_file_path: string;
}

export interface PipelineTrainEvaluateResponse {
  upload_id: string;
  pipeline: PipelineResponse;
  training: Record<string, unknown>;
  evaluation: Record<string, unknown>;
}

export interface RubricItem {
  criterion: string;
  description: string;
  weight: number;
}

export interface ModelInferenceRequest {
  upload_id: string;
  reference_answer: string;
  answer: string;
  rubric: RubricItem[];
  task?: string;
  model_name?: string;
}

export interface ModelInferenceResponse {
  upload_id: string;
  score: number | null;
  reasoning: string;
}

export interface UploadModelResponse {
  upload_id: string;
  repo_id: string;
  repo_url: string;
}

export interface FullRunParams {
  file: File;
  training_config?: File | null;
  base_model_name: string;
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
  seed: number;
  experiment_name?: string;
  num_train_epochs: number;
  learning_rate: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  use_cohen_kappa: boolean;
  use_bertscore: boolean;
}

export interface HuggingFaceUploadParams {
  upload_id: string;
  hf_token: string;
  hf_username: string;
  dataset_name?: string;
  private: boolean;
}

// ── Helpers ──

function getBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Server error: ${response.status}`;
    try {
      const err = await response.json();
      if (Array.isArray(err.detail)) {
        message = err.detail.map((d: { msg: string }) => d.msg).join(', ');
      } else if (typeof err.detail === 'string') {
        message = err.detail;
      }
    } catch { /* ignore */ }
    throw new Error(message);
  }
  return response.json();
}

// ── API Calls ──

export async function runFullPipeline(params: FullRunParams): Promise<PipelineTrainEvaluateResponse> {
  const fd = new FormData();
  fd.append('file', params.file);
  if (params.training_config) fd.append('training_config', params.training_config);
  fd.append('base_model_name', params.base_model_name);
  fd.append('train_ratio', String(params.train_ratio));
  fd.append('val_ratio', String(params.val_ratio));
  fd.append('test_ratio', String(params.test_ratio));
  fd.append('seed', String(params.seed));
  if (params.experiment_name) fd.append('experiment_name', params.experiment_name);
  fd.append('num_train_epochs', String(params.num_train_epochs));
  fd.append('learning_rate', String(params.learning_rate));
  fd.append('per_device_train_batch_size', String(params.per_device_train_batch_size));
  fd.append('gradient_accumulation_steps', String(params.gradient_accumulation_steps));
  fd.append('use_cohen_kappa', String(params.use_cohen_kappa));
  fd.append('use_bertscore', String(params.use_bertscore));

  const res = await fetch(`${getBaseUrl()}/pipeline/full-run`, { method: 'POST', body: fd });
  return handleResponse(res);
}

export async function tryModel(params: ModelInferenceRequest): Promise<ModelInferenceResponse> {
  const body: Record<string, unknown> = {
    upload_id: params.upload_id,
    reference_answer: params.reference_answer,
    answer: params.answer,
    rubric: params.rubric,
  };
  if (params.task) body.task = params.task;
  if (params.model_name) body.model_name = params.model_name;

  const res = await fetch(`${getBaseUrl()}/evaluation/try-model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

export async function uploadToHuggingFace(params: HuggingFaceUploadParams): Promise<UploadModelResponse> {
  const body = new URLSearchParams();
  body.append('upload_id', params.upload_id);
  body.append('hf_token', params.hf_token);
  body.append('hf_username', params.hf_username);
  if (params.dataset_name) body.append('dataset_name', params.dataset_name);
  body.append('private', String(params.private));

  const res = await fetch(`${getBaseUrl()}/upload/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });
  return handleResponse(res);
}
