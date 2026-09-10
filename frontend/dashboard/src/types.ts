export type NodeRecord = {
  node_id: string
  name?: string | null
  lat?: number | null
  lon?: number | null
  metadata?: Record<string, unknown>
}

export type RoadRecord = {
  road_id: string
  from_node: string
  to_node: string
  distance_km: number
  speed_limit_kmph: number
  capacity_vph: number
  free_flow_time_min?: number | null
  is_closed: boolean
  metadata?: Record<string, unknown>
}

export type NetworkResponse = {
  name: string
  nodes: NodeRecord[]
  roads: RoadRecord[]
}

export type TrafficMetric = {
  road_id: string
  expected_flow: number
  hourly_flow: number | null
  capacity_vph: number
  utilization_ratio: number
  free_flow_time_minutes: number
  estimated_travel_time_minutes: number
  congestion_score: number
  congestion_level:
    | 'FREE'
    | 'MODERATE'
    | 'HEAVY'
    | 'SEVERE'
  time_window_start?: string | null
  time_window_end?: string | null
}

export type TrafficResponse = {
  metrics: TrafficMetric[]
  evaluated_roads_count: number
  time_window_start?: string | null
  time_window_end?: string | null
}

export type TrajectoryRoute = {
  nodes: string[]
  probability: number
  metadata?: Record<string, unknown>
}

export type Trajectory = {
  track_id: string
  origin_node: string
  destination_node: string
  vehicle_weight: number
  candidate_routes: TrajectoryRoute[]
  time_window_start?: string | number | null
  time_window_end?: string | number | null
  metadata?: Record<string, unknown>
}

export type TrajectoryResponse = {
  trajectories: Trajectory[]
}

export type HealthResponse = {
  status: string
  engine: string
}