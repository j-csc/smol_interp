// API Request Types
export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  top_k?: number;
}

export interface SteeringConfig {
  steering_feature: number;
  max_act: number;
  steering_strength: number;
}

export interface SteerRequest {
  prompt: string;
  steering_configs: SteeringConfig[];
  max_new_tokens?: number;
}

// API Response Types
export interface FeatureExplanation {
  feature_idx: number;
  activation: number;
  explanation: string;
}

export interface TokenExplanation {
  token: string;
  features: FeatureExplanation[];
}

export interface GenerateResponse {
  prompt: string;
  full_text: string;
  input_token_explanations: TokenExplanation[];
  output_token_explanations: TokenExplanation[];
  feature_max_activations: Record<number, number>;
}

export interface SteerResponse {
  prompt: string;
  full_text: string;
}

export interface ModelInfo {
  model_name: string;
  n_layers: number;
  d_model: number;
}

export interface HealthResponse {
  status: string;
}
