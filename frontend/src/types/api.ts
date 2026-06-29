export interface GenerateRequest {
  prompt: string;
  temperature?: number;
  max_tokens?: number;
}

export interface SSEToken {
  token: string;
  is_end: boolean;
  model: 'base' | 'finetuned';
}

export interface MetricResult {
  model: string;
  rouge1?: number;
  rouge2?: number;
  rougeL?: number;
  perplexity?: string | number;
  avg_response_length?: string | number;
}

export interface MetricsResponse {
  results: MetricResult[];
  eval_samples?: number;
  note?: string;
}

export interface LossData {
  step: number;
  loss: number | null;
  epoch?: number;
  learning_rate?: string | number;
}

export interface LossDataResponse {
  data: LossData[];
  note?: string;
}

export interface AdapterResponse {
  adapters: string[];
}

export interface ValidateRequest {
  code: string;
}

export interface ValidateResponse {
  valid: boolean;
  error?: string;
  errorLine?: number;
}
