import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Car,
  CheckCircle2,
  ChevronRight,
  Clock3,
  GitBranch,
  Gauge,
  Layers3,
  Map as MapIcon,
  Network,
  Play,
  Route,
  ShieldCheck,
  TriangleAlert,
  XCircle,
  Zap,
} from 'lucide-react'
import {
  Canvas,
} from '@react-three/fiber'
import {
  Grid,
  OrbitControls,
  PerspectiveCamera,
} from '@react-three/drei'

import CityMap from './components/CityMap'
import type {
  HealthResponse,
  NetworkResponse,
  TrafficResponse,
  Trajectory,
  TrajectoryResponse,
} from './types'

type OdPair = {
  origin: string
  destination: string
  demand: number
  time_window_start?: string | null
  time_window_end?: string | null
}

type OdResponse = {
  total_demand: number
  trajectory_count: number
  pairs: OdPair[]
  route_demands: Array<{
    route_nodes: string[]
    road_ids: string[]
    demand: number
    time_window_start?: string | null
    time_window_end?: string | null
  }>
}

type Bottleneck = {
  road_id: string
  severity: string
  utilization_ratio: number
  congestion_score: number
  hourly_flow: number
  capacity_vph: number
  delay_minutes: number
  bottleneck_score: number
  signals: string[]
}

type BottleneckResponse = {
  bottlenecks: Bottleneck[]
}

type AnomalyResponse = {
  baseline_snapshot_id: string
  current_snapshot_id: string
  road_evidence: Array<{
    road_id: string
    baseline_hourly_flow?: number | null
    current_hourly_flow?: number | null
    absolute_change?: number | null
    relative_change?: number | null
    status?: string
  }>
  od_divergence?: number
  route_divergence?: number
  hhi_delta?: number
  bottleneck_changes?: unknown[]
  anomaly_events?: Array<{
    severity?: string
    signal?: string
    message?: string
  }>
  spatial_regions?: unknown[]
}

type SimulationResult = {
  scenario: {
    scenario_id: string
    name: string
    description: string
    closed_road_ids: string[]
    capacity_modifications_vph: Record<string, number>
    speed_modifications_kmph: Record<string, number>
  }
  baseline_summary: Record<string, unknown>
  scenario_summary: Record<string, unknown>
  road_impacts: Array<{
    road_id: string
    baseline_hourly_flow: number
    scenario_hourly_flow: number
    flow_delta: number
    baseline_utilization: number
    scenario_utilization: number
    utilization_delta: number
    baseline_congestion_level?: string | null
    scenario_congestion_level?: string | null
    travel_time_delta_minutes: number
    additional_flow: number
    became_new_bottleneck: boolean
  }>
  od_impacts: Array<{
    origin: string
    destination: string
    demand: number
    baseline_route: string[]
    scenario_route: string[]
    route_changed: boolean
    baseline_travel_time_minutes?: number | null
    scenario_travel_time_minutes?: number | null
    travel_time_delta_minutes?: number | null
    status: string
  }>
  unroutable: Array<{
    origin: string
    destination: string
    demand: number
    reason: string
  }>
  decision: {
    scenario_id: string
    feasible: boolean
    affected_road_count: number
    rerouted_demand: number
    newly_congested_roads: string[]
    relieved_roads: string[]
    total_travel_time_delta_minutes: number
    network_congestion_delta: number
    unroutable_demand: number
    recommendation_class: 'FAVORABLE' | 'NEUTRAL' | 'UNFAVORABLE'
    explanation: string
  }
}

type SimulationMode = 'closure' | 'capacity' | 'speed'

const API = '/api'

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`)

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`)
  }

  return response.json()
}

async function postJson<T>(
  path: string,
  payload: unknown,
): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    let message = `Simulation failed: ${response.status}`

    try {
      const body = await response.json()

      if (typeof body?.detail === 'string') {
        message = body.detail
      }
    } catch {
      // Keep default error message.
    }

    throw new Error(message)
  }

  return response.json()
}

function formatNumber(value: number, digits = 1) {
  if (!Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

function formatPercent(value: number, digits = 1) {
  if (!Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

function formatDelta(value: number, digits = 3) {
  if (!Number.isFinite(value)) return '—'

  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function NetworkScene({
  network,
  traffic,
}: {
  network: NetworkResponse
  traffic: TrafficResponse
}) {
  /*
   * Explicit Map type fixes TypeScript inference:
   * road_id -> traffic metric
   */
  const trafficByRoad = useMemo(
    () =>
      new Map<
        string,
        TrafficResponse['metrics'][number]
      >(
        traffic.metrics.map((metric) => [
          metric.road_id,
          metric,
        ]),
      ),
    [traffic.metrics],
  )

  const coordinates = network.nodes
    .filter(
      (node) =>
        typeof node.lat === 'number' &&
        typeof node.lon === 'number',
    )
    .map((node) => ({
      ...node,
      x: ((node.lon ?? 0) - 77.55) * 100,
      z: -((node.lat ?? 0) - 28.58) * 100,
    }))

  /*
   * Explicit Map type fixes the second TypeScript
   * inference issue:
   * node_id -> coordinate node
   */
  const nodeMap = new Map<
    string,
    (typeof coordinates)[number]
  >(
    coordinates.map((node) => [
      node.node_id,
      node,
    ]),
  )

  return (
    <>
      <ambientLight intensity={1.2} />

      <directionalLight
        position={[20, 30, 20]}
        intensity={2}
      />

      {network.roads.map((road) => {
        const from = nodeMap.get(road.from_node)
        const to = nodeMap.get(road.to_node)

        if (!from || !to) return null

        const metric = trafficByRoad.get(
          road.road_id,
        )

        const x =
          (from.x + to.x) / 2

        const y = 0.2

        const z =
          (from.z + to.z) / 2

        const length = Math.sqrt(
          (to.x - from.x) ** 2 +
            (to.z - from.z) ** 2,
        )

        const rotation = Math.atan2(
          to.z - from.z,
          to.x - from.x,
        )

        const congestion =
          metric?.congestion_level ?? 'FREE'

        const height =
          congestion === 'SEVERE'
            ? 1.2
            : congestion === 'HEAVY'
              ? 0.9
              : congestion === 'MODERATE'
                ? 0.65
                : 0.4

        return (
          <mesh
            key={road.road_id}
            position={[x, y, z]}
            rotation={[
              0,
              -rotation,
              0,
            ]}
          >
            <boxGeometry
              args={[
                Math.max(length, 0.8),
                height,
                0.25,
              ]}
            />

            <meshStandardMaterial
              emissiveIntensity={0.6}
              transparent
              opacity={
                road.is_closed
                  ? 0.25
                  : 0.9
              }
            />
          </mesh>
        )
      })}

      {coordinates.map((node) => (
        <mesh
          key={node.node_id}
          position={[
            node.x,
            0.8,
            node.z,
          ]}
        >
          <sphereGeometry
            args={[0.55, 16, 16]}
          />

          <meshStandardMaterial
            emissiveIntensity={1}
          />
        </mesh>
      ))}
    </>
  )
}

function App() {
  const [network, setNetwork] =
    useState<NetworkResponse | null>(null)

  const [traffic, setTraffic] =
    useState<TrafficResponse | null>(null)

  const [od, setOd] =
    useState<OdResponse | null>(null)

  const [bottlenecks, setBottlenecks] =
    useState<BottleneckResponse | null>(null)

  const [anomalies, setAnomalies] =
    useState<AnomalyResponse | null>(null)

  const [trajectories, setTrajectories] =
    useState<Trajectory[]>([])

  const [health, setHealth] =
    useState<HealthResponse | null>(null)

  const [loading, setLoading] =
    useState(true)

  const [error, setError] =
    useState<string | null>(null)

  // ---------------------------------------------------------------------------
  // Simulation state
  // ---------------------------------------------------------------------------

  const [simulationMode, setSimulationMode] =
    useState<SimulationMode>('closure')

  const [simulationRoadId, setSimulationRoadId] =
    useState('')

  const [simulationValue, setSimulationValue] =
    useState('')

  const [simulationLoading, setSimulationLoading] =
    useState(false)

  const [simulationError, setSimulationError] =
    useState<string | null>(null)

  const [simulationResult, setSimulationResult] =
    useState<SimulationResult | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      try {
        setLoading(true)
        setError(null)

        const [
          healthData,
          networkData,
          trafficData,
          odData,
          bottleneckData,
          anomalyData,
          trajectoryData,
        ] = await Promise.all([
          fetchJson<HealthResponse>(
            '/health',
          ),
          fetchJson<NetworkResponse>(
            '/network',
          ),
          fetchJson<TrafficResponse>(
            '/traffic',
          ),
          fetchJson<OdResponse>(
            '/analytics/od',
          ),
          fetchJson<BottleneckResponse>(
            '/analytics/bottlenecks',
          ),
          fetchJson<AnomalyResponse>(
            '/anomalies',
          ),
          fetchJson<TrajectoryResponse>(
            '/trajectories',
          ),
        ])

        if (cancelled) return

        setHealth(healthData)
        setNetwork(networkData)
        setTraffic(trafficData)
        setOd(odData)
        setBottlenecks(
          bottleneckData,
        )
        setAnomalies(anomalyData)
        setTrajectories(
          trajectoryData.trajectories,
        )

        if (
          !simulationRoadId &&
          networkData.roads.length
        ) {
          setSimulationRoadId(
            networkData.roads[0].road_id,
          )
        }
      } catch (err) {
        if (cancelled) return

        setError(
          err instanceof Error
            ? err.message
            : 'Failed to load dashboard',
        )
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    loadDashboard()

    return () => {
      cancelled = true
    }
  }, [])

  const trafficSummary = useMemo(() => {
    if (!traffic?.metrics.length) {
      return {
        averageUtilization: 0,
        severeRoads: 0,
        heavyRoads: 0,
      }
    }

    const averageUtilization =
      traffic.metrics.reduce(
        (sum, metric) =>
          sum + metric.utilization_ratio,
        0,
      ) / traffic.metrics.length

    return {
      averageUtilization,
      severeRoads:
        traffic.metrics.filter(
          (metric) =>
            metric.congestion_level ===
            'SEVERE',
        ).length,
      heavyRoads:
        traffic.metrics.filter(
          (metric) =>
            metric.congestion_level ===
            'HEAVY',
        ).length,
    }
  }, [traffic])

  const topOdPairs = useMemo(
    () =>
      [...(od?.pairs ?? [])]
        .sort(
          (a, b) =>
            b.demand - a.demand,
        )
        .slice(0, 5),
    [od],
  )

  const significantRoadChanges =
    useMemo(
      () =>
        [...(anomalies?.road_evidence ?? [])]
          .filter(
            (item) =>
              typeof item.relative_change ===
              'number',
          )
          .sort(
            (a, b) =>
              Math.abs(
                b.relative_change ?? 0,
              ) -
              Math.abs(
                a.relative_change ?? 0,
              ),
          )
          .slice(0, 5),
      [anomalies],
    )

  const runSimulation = async () => {
    if (!simulationRoadId) {
      setSimulationError(
        'Select a road before running the simulation.',
      )
      return
    }

    const numericValue =
      Number(simulationValue)

    if (
      (simulationMode === 'capacity' ||
        simulationMode === 'speed') &&
      (!Number.isFinite(numericValue) ||
        numericValue <= 0)
    ) {
      setSimulationError(
        'Enter a positive value for the selected intervention.',
      )
      return
    }

    setSimulationLoading(true)
    setSimulationError(null)

    try {
      const scenarioId =
        `ui-${Date.now()}`

      const payload = {
        scenario_id: scenarioId,
        name:
          simulationMode === 'closure'
            ? `Close ${simulationRoadId}`
            : simulationMode ===
                'capacity'
              ? `Capacity reduction on ${simulationRoadId}`
              : `Speed reduction on ${simulationRoadId}`,
        description:
          simulationMode === 'closure'
            ? `Counterfactual closure of ${simulationRoadId}`
            : simulationMode ===
                'capacity'
              ? `Counterfactual capacity intervention on ${simulationRoadId}`
              : `Counterfactual speed intervention on ${simulationRoadId}`,
        closed_road_ids:
          simulationMode === 'closure'
            ? [simulationRoadId]
            : [],
        capacity_modifications_vph:
          simulationMode ===
          'capacity'
            ? {
                [simulationRoadId]:
                  numericValue,
              }
            : {},
        speed_modifications_kmph:
          simulationMode === 'speed'
            ? {
                [simulationRoadId]:
                  numericValue,
              }
            : {},
      }

      const result =
        await postJson<SimulationResult>(
          '/simulation',
          payload,
        )

      setSimulationResult(result)
    } catch (err) {
      setSimulationError(
        err instanceof Error
          ? err.message
          : 'Simulation failed',
      )
      setSimulationResult(null)
    } finally {
      setSimulationLoading(false)
    }
  }

  const selectedSimulationRoad =
    network?.roads.find(
      (road) =>
        road.road_id ===
        simulationRoadId,
    )

  if (loading) {
    return (
      <div className="app-shell loading-shell">
        <div className="loading-card">
          <Activity size={20} />
          <span>
            INITIALIZING URBANTRACKAI
          </span>
        </div>
      </div>
    )
  }

  if (error || !network || !traffic) {
    return (
      <div className="app-shell loading-shell">
        <div className="loading-card error-card">
          <TriangleAlert size={20} />
          <span>
            {error ??
              'Dashboard data unavailable'}
          </span>
        </div>
      </div>
    )
  }

  const decision =
    simulationResult?.decision

  const recommendation =
    decision?.recommendation_class

  return (
    <div className="app-shell">
      {/* ------------------------------------------------------------------ */}
      {/* HEADER                                                             */}
      {/* ------------------------------------------------------------------ */}

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <Network size={21} />
          </div>

          <div>
            <div className="brand-name">
              URBANTRACK<span>AI</span>
            </div>

            <div className="brand-subtitle">
              CITY-WIDE MOBILITY INTELLIGENCE
            </div>
          </div>
        </div>

        <div className="topbar-status">
          <div className="status-dot" />
          <span>
            {health?.status === 'ok'
              ? 'ENGINE OPERATIONAL'
              : 'ENGINE DEGRADED'}
          </span>

          <span className="status-divider" />

          <span>
            {network.nodes.length} NODES
          </span>

          <span>
            {network.roads.length} ROADS
          </span>
        </div>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* HERO / SYSTEM STATUS                                               */}
      {/* ------------------------------------------------------------------ */}

      <section className="hero-section">
        <div>
          <div className="eyebrow">
            URBAN MOBILITY COMMAND CENTER
          </div>

          <h1>
            CITY TRAFFIC
            <br />
            <span>INTELLIGENCE</span>
          </h1>

          <p className="hero-copy">
            From distributed vehicle observations
            to probabilistic journeys,
            network-level anomalies and
            counterfactual decisions.
          </p>
        </div>

        <div className="hero-metrics">
          <div className="hero-metric">
            <span>TRAJECTORIES</span>
            <strong>
              {trajectories.length}
            </strong>
          </div>

          <div className="hero-metric">
            <span>TOTAL DEMAND</span>
            <strong>
              {formatNumber(
                od?.total_demand ?? 0,
              )}
            </strong>
          </div>

          <div className="hero-metric">
            <span>AVG UTILIZATION</span>
            <strong>
              {formatPercent(
                trafficSummary.averageUtilization,
              )}
            </strong>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* KPI GRID                                                           */}
      {/* ------------------------------------------------------------------ */}

      <section className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">
            <Car size={18} />
          </div>

          <div>
            <span>ACTIVE VEHICLES</span>
            <strong>
              {trajectories.length}
            </strong>
            <small>
              inferred trajectories
            </small>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">
            <Gauge size={18} />
          </div>

          <div>
            <span>NETWORK UTILIZATION</span>
            <strong>
              {formatPercent(
                trafficSummary.averageUtilization,
              )}
            </strong>
            <small>
              flow-weighted network state
            </small>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">
            <AlertTriangle size={18} />
          </div>

          <div>
            <span>HEAVY / SEVERE ROADS</span>
            <strong>
              {trafficSummary.heavyRoads +
                trafficSummary.severeRoads}
            </strong>
            <small>
              congestion pressure points
            </small>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">
            <Route size={18} />
          </div>

          <div>
            <span>OD DEMAND</span>
            <strong>
              {formatNumber(
                od?.total_demand ?? 0,
              )}
            </strong>
            <small>
              origin-destination trips
            </small>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* 3D NETWORK                                                         */}
      {/* ------------------------------------------------------------------ */}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">
              NETWORK MODEL
            </div>

            <h2>
              CITY MOBILITY GRAPH
            </h2>
          </div>

          <div className="section-badge">
            <Layers3 size={14} />
            3D NETWORK VIEW
          </div>
        </div>

        <div className="network-3d-card">
          <Canvas>
            <PerspectiveCamera
              makeDefault
              position={[
                0,
                32,
                34,
              ]}
            />

            <NetworkScene
              network={network}
              traffic={traffic}
            />

            <Grid
              args={[80, 80]}
              cellSize={2}
              cellThickness={0.5}
              sectionSize={10}
              sectionThickness={1}
              fadeDistance={70}
              fadeStrength={1}
            />

            <OrbitControls
              enableDamping
              dampingFactor={0.08}
            />
          </Canvas>

          <div className="network-overlay">
            <div>
              <span>NODES</span>
              <strong>
                {network.nodes.length}
              </strong>
            </div>

            <div>
              <span>EDGES</span>
              <strong>
                {network.roads.length}
              </strong>
            </div>

            <div>
              <span>EVALUATED</span>
              <strong>
                {traffic.evaluated_roads_count}
              </strong>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* GIS MAP                                                            */}
      {/* ------------------------------------------------------------------ */}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">
              GEOSPATIAL INTELLIGENCE
            </div>

            <h2>
              LIVE MOBILITY MAP
            </h2>
          </div>

          <div className="section-badge">
            <MapIcon size={14} />
            GIS LAYER
          </div>
        </div>

        <CityMap
          network={network}
          traffic={traffic.metrics}
          trajectories={trajectories}
        />
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* ANOMALY INTELLIGENCE                                               */}
      {/* ------------------------------------------------------------------ */}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">
              NETWORK ANOMALY INTELLIGENCE
            </div>

            <h2>
              MOBILITY STATE MONITOR
            </h2>
          </div>

          <div className="section-badge">
            <BrainCircuit size={14} />
            MULTI-SIGNAL DETECTOR
          </div>
        </div>

        <div className="anomaly-grid">
          <div className="anomaly-main">
            <div className="anomaly-status-row">
              <div
                className={`anomaly-status ${
                  anomalies?.anomaly_events?.length
                    ? 'warning'
                    : 'normal'
                }`}
              >
                {anomalies?.anomaly_events?.length ? (
                  <TriangleAlert
                    size={18}
                  />
                ) : (
                  <ShieldCheck
                    size={18}
                  />
                )}

                <div>
                  <span>
                    DETECTOR ASSESSMENT
                  </span>

                  <strong>
                    {anomalies?.anomaly_events
                      ?.length
                      ? 'ANOMALY SIGNAL DETECTED'
                      : 'NO CONSENSUS ANOMALY'}
                  </strong>
                </div>
              </div>

              <div className="evidence-count">
                <span>
                  EVIDENCE SIGNALS
                </span>

                <strong>
                  {anomalies
                    ?.anomaly_events
                    ?.length ?? 0}
                </strong>
              </div>
            </div>

            <div className="anomaly-stats">
              <div>
                <span>
                  BASELINE STATE
                </span>

                <strong>
                  {anomalies?.baseline_snapshot_id ??
                    '—'}
                </strong>
              </div>

              <div>
                <span>
                  CURRENT STATE
                </span>

                <strong>
                  {anomalies?.current_snapshot_id ??
                    '—'}
                </strong>
              </div>

              <div>
                <span>
                  OD DIVERGENCE
                </span>

                <strong>
                  {formatNumber(
                    anomalies?.od_divergence ??
                      0,
                    4,
                  )}
                </strong>

                <small>
                  Jensen–Shannon divergence
                </small>
              </div>

              <div>
                <span>
                  ROUTE DIVERGENCE
                </span>

                <strong>
                  {formatNumber(
                    anomalies?.route_divergence ??
                      0,
                    4,
                  )}
                </strong>

                <small>
                  route redistribution
                </small>
              </div>

              <div>
                <span>
                  HHI DELTA
                </span>

                <strong>
                  {formatDelta(
                    anomalies?.hhi_delta ??
                      0,
                    4,
                  )}
                </strong>

                <small>
                  network concentration
                </small>
              </div>

              <div>
                <span>
                  BOTTLENECK SHIFT
                </span>

                <strong>
                  {anomalies?.bottleneck_changes
                    ?.length ?? 0}
                </strong>

                <small>
                  detected changes
                </small>
              </div>
            </div>

            <div className="anomaly-explanation">
              <span>
                DETECTOR ASSESSMENT
              </span>

              <p>
                {anomalies?.anomaly_events
                  ?.length
                  ? 'Multiple mobility signals have crossed configured detection thresholds and require investigation.'
                  : 'Current mobility deviations remain below configured detection thresholds.'}
              </p>

              <small>
                The detector compares the current
                mobility state against a synthetic
                normal baseline using flow, OD
                demand, route redistribution,
                network concentration and
                bottleneck evidence.
              </small>
            </div>
          </div>

          <div className="anomaly-side">
            <div className="side-panel-title">
              SIGNIFICANT ROAD CHANGES
            </div>

            <div className="change-list">
              {significantRoadChanges.map(
                (item) => {
                  const change =
                    item.relative_change ??
                    0

                  const positive =
                    change >= 0

                  return (
                    <div
                      className="change-row"
                      key={item.road_id}
                    >
                      <span>
                        {item.road_id}
                      </span>

                      <strong
                        className={
                          positive
                            ? 'positive'
                            : 'negative'
                        }
                      >
                        {positive ? (
                          <ArrowUpRight
                            size={13}
                          />
                        ) : (
                          <ArrowDownRight
                            size={13}
                          />
                        )}

                        {Math.abs(
                          change * 100,
                        ).toFixed(1)}
                        %
                      </strong>
                    </div>
                  )
                },
              )}

              {!significantRoadChanges.length && (
                <div className="empty-state">
                  No material road changes.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* ANALYTICS GRID                                                     */}
      {/* ------------------------------------------------------------------ */}

      <section className="two-column-grid">
        <div className="dashboard-panel">
          <div className="panel-heading">
            <div>
              <span>
                ORIGIN / DESTINATION
              </span>

              <h3>
                TOP MOBILITY PAIRS
              </h3>
            </div>

            <BarChart3 size={18} />
          </div>

          <div className="table-list">
            {topOdPairs.map(
              (pair, index) => (
                <div
                  className="table-row"
                  key={`${pair.origin}-${pair.destination}`}
                >
                  <span className="rank">
                    {String(
                      index + 1,
                    ).padStart(2, '0')}
                  </span>

                  <div className="route-cell">
                    <strong>
                      {pair.origin}
                      <ChevronRight
                        size={13}
                      />
                      {pair.destination}
                    </strong>

                    <small>
                      demand flow
                    </small>
                  </div>

                  <b>
                    {formatNumber(
                      pair.demand,
                    )}
                  </b>
                </div>
              ),
            )}
          </div>
        </div>

        <div className="dashboard-panel">
          <div className="panel-heading">
            <div>
              <span>
                TRAFFIC PRESSURE
              </span>

              <h3>
                BOTTLENECK INTELLIGENCE
              </h3>
            </div>

            <Gauge size={18} />
          </div>

          <div className="table-list">
            {bottlenecks?.bottlenecks
              ?.slice(0, 5)
              .map((item) => (
                <div
                  className="table-row"
                  key={item.road_id}
                >
                  <div className="route-cell">
                    <strong>
                      {item.road_id}
                    </strong>

                    <small>
                      {item.severity}
                    </small>
                  </div>

                  <div className="mini-bar">
                    <span
                      style={{
                        width: `${Math.min(
                          item.utilization_ratio *
                            100,
                          100,
                        )}%`,
                      }}
                    />
                  </div>

                  <b>
                    {formatPercent(
                      item.utilization_ratio,
                    )}
                  </b>
                </div>
              ))}

            {!bottlenecks?.bottlenecks?.length && (
              <div className="empty-state">
                No bottlenecks detected.
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* DECISION LAYER — WHAT-IF SIMULATION                                */}
      {/* ------------------------------------------------------------------ */}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">
              DECISION INTELLIGENCE
            </div>

            <h2>
              WHAT-IF SIMULATION
            </h2>
          </div>

          <div className="section-badge">
            <Zap size={14} />
            COUNTERFACTUAL ENGINE
          </div>
        </div>

        <div
          className="simulation-shell"
          style={{
            border:
              '1px solid rgba(125, 249, 255, 0.18)',
            borderRadius: 16,
            background:
              'linear-gradient(135deg, rgba(5,18,25,.98), rgba(7,25,34,.94))',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateColumns:
                'minmax(280px, 0.85fr) minmax(360px, 1.15fr)',
            }}
          >
            {/* Controls */}

            <div
              style={{
                padding: 24,
                borderRight:
                  '1px solid rgba(255,255,255,.07)',
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 800,
                  letterSpacing:
                    '0.14em',
                  color: '#7df9ff',
                  marginBottom: 8,
                }}
              >
                COUNTERFACTUAL SCENARIO
              </div>

              <h3
                style={{
                  margin: '0 0 8px',
                  fontSize: 24,
                  color: '#fff',
                }}
              >
                Change the network.
              </h3>

              <p
                style={{
                  margin:
                    '0 0 22px',
                  color: '#78939e',
                  fontSize: 12,
                  lineHeight: 1.6,
                }}
              >
                Select a road intervention and
                let the mobility engine estimate
                rerouting, travel-time impact and
                network consequences.
              </p>

              <div
                style={{
                  display: 'grid',
                  gap: 16,
                }}
              >
                <label
                  style={{
                    display: 'grid',
                    gap: 7,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 800,
                      letterSpacing:
                        '0.1em',
                      color: '#8ca8b3',
                    }}
                  >
                    INTERVENTION
                  </span>

                  <select
                    value={
                      simulationMode
                    }
                    onChange={(event) => {
                      setSimulationMode(
                        event.target
                          .value as SimulationMode,
                      )
                      setSimulationResult(
                        null,
                      )
                      setSimulationError(
                        null,
                      )
                    }}
                    style={{
                      width: '100%',
                      boxSizing:
                        'border-box',
                      padding:
                        '12px 13px',
                      border:
                        '1px solid rgba(125,249,255,.2)',
                      borderRadius: 9,
                      background:
                        '#081820',
                      color: '#eafcff',
                      outline: 'none',
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    <option value="closure">
                      Close road
                    </option>

                    <option value="capacity">
                      Reduce capacity
                    </option>

                    <option value="speed">
                      Reduce speed limit
                    </option>
                  </select>
                </label>

                <label
                  style={{
                    display: 'grid',
                    gap: 7,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 800,
                      letterSpacing:
                        '0.1em',
                      color: '#8ca8b3',
                    }}
                  >
                    TARGET ROAD
                  </span>

                  <select
                    value={
                      simulationRoadId
                    }
                    onChange={(event) => {
                      setSimulationRoadId(
                        event.target.value,
                      )
                      setSimulationResult(
                        null,
                      )
                      setSimulationError(
                        null,
                      )
                    }}
                    style={{
                      width: '100%',
                      boxSizing:
                        'border-box',
                      padding:
                        '12px 13px',
                      border:
                        '1px solid rgba(125,249,255,.2)',
                      borderRadius: 9,
                      background:
                        '#081820',
                      color: '#eafcff',
                      outline: 'none',
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    {network.roads.map(
                      (road) => (
                        <option
                          key={
                            road.road_id
                          }
                          value={
                            road.road_id
                          }
                        >
                          {road.road_id} ·{' '}
                          {
                            road.from_node
                          } →{' '}
                          {
                            road.to_node
                          }
                        </option>
                      ),
                    )}
                  </select>
                </label>

                {simulationMode !==
                  'closure' && (
                  <label
                    style={{
                      display: 'grid',
                      gap: 7,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 800,
                        letterSpacing:
                          '0.1em',
                        color: '#8ca8b3',
                      }}
                    >
                      {simulationMode ===
                      'capacity'
                        ? 'NEW CAPACITY · VPH'
                        : 'NEW SPEED LIMIT · KM/H'}
                    </span>

                    <input
                      type="number"
                      min="0.01"
                      step="1"
                      value={
                        simulationValue
                      }
                      onChange={(event) => {
                        setSimulationValue(
                          event.target
                            .value,
                        )
                        setSimulationError(
                          null,
                        )
                      }}
                      placeholder={
                        simulationMode ===
                        'capacity'
                          ? 'e.g. 800'
                          : 'e.g. 25'
                      }
                      style={{
                        width: '100%',
                        boxSizing:
                          'border-box',
                        padding:
                          '12px 13px',
                        border:
                          '1px solid rgba(125,249,255,.2)',
                        borderRadius: 9,
                        background:
                          '#081820',
                        color:
                          '#eafcff',
                        outline: 'none',
                        fontSize: 12,
                        fontWeight: 700,
                      }}
                    />
                  </label>
                )}

                {selectedSimulationRoad && (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns:
                        '1fr 1fr',
                      gap: 8,
                    }}
                  >
                    <div
                      style={{
                        padding: 11,
                        borderRadius: 9,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          fontSize: 9,
                          color:
                            '#6f8994',
                          marginBottom: 4,
                        }}
                      >
                        CAPACITY
                      </span>

                      <strong
                        style={{
                          color:
                            '#fff',
                          fontSize: 14,
                        }}
                      >
                        {
                          selectedSimulationRoad.capacity_vph
                        }
                        <small>
                          {' '}
                          vph
                        </small>
                      </strong>
                    </div>

                    <div
                      style={{
                        padding: 11,
                        borderRadius: 9,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          fontSize: 9,
                          color:
                            '#6f8994',
                          marginBottom: 4,
                        }}
                      >
                        SPEED LIMIT
                      </span>

                      <strong
                        style={{
                          color:
                            '#fff',
                          fontSize: 14,
                        }}
                      >
                        {
                          selectedSimulationRoad.speed_limit_kmph
                        }
                        <small>
                          {' '}
                          km/h
                        </small>
                      </strong>
                    </div>
                  </div>
                )}

                {simulationError && (
                  <div
                    style={{
                      display: 'flex',
                      gap: 8,
                      alignItems:
                        'flex-start',
                      padding: 11,
                      borderRadius: 9,
                      border:
                        '1px solid rgba(255,91,91,.22)',
                      background:
                        'rgba(255,91,91,.06)',
                      color: '#ff8f8f',
                      fontSize: 11,
                      lineHeight: 1.5,
                    }}
                  >
                    <AlertTriangle
                      size={14}
                    />

                    <span>
                      {
                        simulationError
                      }
                    </span>
                  </div>
                )}

                <button
                  type="button"
                  onClick={
                    runSimulation
                  }
                  disabled={
                    simulationLoading
                  }
                  style={{
                    display: 'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center',
                    gap: 9,
                    width: '100%',
                    padding:
                      '13px 16px',
                    border: 'none',
                    borderRadius: 9,
                    background:
                      simulationLoading
                        ? '#17333d'
                        : '#7df9ff',
                    color:
                      simulationLoading
                        ? '#73909b'
                        : '#031116',
                    cursor:
                      simulationLoading
                        ? 'wait'
                        : 'pointer',
                    fontSize: 11,
                    fontWeight: 900,
                    letterSpacing:
                      '0.08em',
                  }}
                >
                  <Play size={14} />

                  {simulationLoading
                    ? 'RUNNING SIMULATION...'
                    : 'RUN SIMULATION'}
                </button>
              </div>
            </div>

            {/* Results */}

            <div
              style={{
                padding: 24,
                minWidth: 0,
              }}
            >
              {!simulationResult ? (
                <div
                  style={{
                    minHeight: 330,
                    display: 'flex',
                    alignItems:
                      'center',
                    justifyContent:
                      'center',
                    textAlign: 'center',
                    padding: 30,
                  }}
                >
                  <div>
                    <div
                      style={{
                        width: 54,
                        height: 54,
                        margin:
                          '0 auto 15px',
                        display: 'grid',
                        placeItems:
                          'center',
                        borderRadius:
                          '50%',
                        background:
                          'rgba(125,249,255,.07)',
                        color:
                          '#7df9ff',
                      }}
                    >
                      <GitBranch
                        size={23}
                      />
                    </div>

                    <strong
                      style={{
                        display:
                          'block',
                        color:
                          '#dffaff',
                        fontSize: 15,
                        marginBottom:
                          7,
                      }}
                    >
                      Simulation ready
                    </strong>

                    <p
                      style={{
                        maxWidth:
                          390,
                        margin: 0,
                        color:
                          '#6f8994',
                        fontSize: 11,
                        lineHeight: 1.6,
                      }}
                    >
                      Choose an intervention
                      and run the counterfactual
                      engine to see how traffic
                      demand redistributes through
                      the city network.
                    </p>
                  </div>
                </div>
              ) : (
                <div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems:
                        'flex-start',
                      justifyContent:
                        'space-between',
                      gap: 16,
                      marginBottom:
                        20,
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontSize: 9,
                          fontWeight: 800,
                          letterSpacing:
                            '0.13em',
                          color:
                            '#7df9ff',
                          marginBottom:
                            5,
                        }}
                      >
                        SIMULATION RESULT
                      </div>

                      <h3
                        style={{
                          margin: 0,
                          color:
                            '#fff',
                          fontSize: 22,
                        }}
                      >
                        {
                          simulationResult
                            .scenario
                            .name
                        }
                      </h3>
                    </div>

                    <div
                      style={{
                        display: 'flex',
                        alignItems:
                          'center',
                        gap: 7,
                        padding:
                          '8px 11px',
                        borderRadius:
                          999,
                        background:
                          recommendation ===
                          'FAVORABLE'
                            ? 'rgba(91,255,170,.09)'
                            : recommendation ===
                                'UNFAVORABLE'
                              ? 'rgba(255,91,91,.09)'
                              : 'rgba(255,209,102,.09)',
                        color:
                          recommendation ===
                          'FAVORABLE'
                            ? '#7dffb4'
                            : recommendation ===
                                'UNFAVORABLE'
                              ? '#ff8f8f'
                              : '#ffd166',
                        fontSize: 10,
                        fontWeight: 900,
                      }}
                    >
                      {recommendation ===
                      'FAVORABLE' ? (
                        <CheckCircle2
                          size={14}
                        />
                      ) : recommendation ===
                        'UNFAVORABLE' ? (
                        <XCircle
                          size={14}
                        />
                      ) : (
                        <AlertTriangle
                          size={14}
                        />
                      )}

                      {
                        recommendation
                      }
                    </div>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns:
                        'repeat(2, minmax(0, 1fr))',
                      gap: 9,
                      marginBottom:
                        16,
                    }}
                  >
                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            5,
                          letterSpacing:
                            '.07em',
                        }}
                      >
                        REROUTED DEMAND
                      </span>

                      <strong
                        style={{
                          color:
                            '#7df9ff',
                          fontSize: 20,
                        }}
                      >
                        {formatNumber(
                          decision
                            ?.rerouted_demand ??
                            0,
                          2,
                        )}
                      </strong>
                    </div>

                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            5,
                          letterSpacing:
                            '.07em',
                        }}
                      >
                        UNROUTABLE DEMAND
                      </span>

                      <strong
                        style={{
                          color:
                            decision?.unroutable_demand
                              ? '#ff8f8f'
                              : '#7dffb4',
                          fontSize: 20,
                        }}
                      >
                        {formatNumber(
                          decision
                            ?.unroutable_demand ??
                            0,
                          2,
                        )}
                      </strong>
                    </div>

                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            5,
                          letterSpacing:
                            '.07em',
                        }}
                      >
                        TRAVEL-TIME DELTA
                      </span>

                      <strong
                        style={{
                          color:
                            (decision
                              ?.total_travel_time_delta_minutes ??
                              0) > 0
                              ? '#ffb0b0'
                              : '#7dffb4',
                          fontSize: 20,
                        }}
                      >
                        {formatDelta(
                          decision
                            ?.total_travel_time_delta_minutes ??
                            0,
                          3,
                        )}
                        <small
                          style={{
                            fontSize: 10,
                            marginLeft: 4,
                          }}
                        >
                          min
                        </small>
                      </strong>
                    </div>

                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.035)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            5,
                          letterSpacing:
                            '.07em',
                        }}
                      >
                        NETWORK CONGESTION
                      </span>

                      <strong
                        style={{
                          color:
                            (decision
                              ?.network_congestion_delta ??
                              0) > 0
                              ? '#ffb0b0'
                              : '#7dffb4',
                          fontSize: 20,
                        }}
                      >
                        {formatDelta(
                          decision
                            ?.network_congestion_delta ??
                            0,
                          4,
                        )}
                      </strong>
                    </div>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns:
                        '1fr 1fr',
                      gap: 9,
                      marginBottom:
                        16,
                    }}
                  >
                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.025)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            7,
                        }}
                      >
                        NEWLY CONGESTED
                      </span>

                      {decision
                        ?.newly_congested_roads
                        ?.length ? (
                        <div
                          style={{
                            display:
                              'flex',
                            flexWrap:
                              'wrap',
                            gap: 5,
                          }}
                        >
                          {decision.newly_congested_roads.map(
                            (road) => (
                              <span
                                key={road}
                                style={{
                                  padding:
                                    '4px 7px',
                                  borderRadius:
                                    5,
                                  background:
                                    'rgba(255,91,91,.1)',
                                  color:
                                    '#ff9b9b',
                                  fontSize:
                                    9,
                                  fontWeight:
                                    800,
                                }}
                              >
                                {road}
                              </span>
                            ),
                          )}
                        </div>
                      ) : (
                        <strong
                          style={{
                            color:
                              '#7dffb4',
                            fontSize: 11,
                          }}
                        >
                          NONE
                        </strong>
                      )}
                    </div>

                    <div
                      style={{
                        padding: 13,
                        borderRadius: 10,
                        background:
                          'rgba(255,255,255,.025)',
                      }}
                    >
                      <span
                        style={{
                          display:
                            'block',
                          color:
                            '#708995',
                          fontSize: 9,
                          marginBottom:
                            7,
                        }}
                      >
                        RELIEVED ROADS
                      </span>

                      {decision
                        ?.relieved_roads
                        ?.length ? (
                        <div
                          style={{
                            display:
                              'flex',
                            flexWrap:
                              'wrap',
                            gap: 5,
                          }}
                        >
                          {decision.relieved_roads.map(
                            (road) => (
                              <span
                                key={road}
                                style={{
                                  padding:
                                    '4px 7px',
                                  borderRadius:
                                    5,
                                  background:
                                    'rgba(125,255,180,.08)',
                                  color:
                                    '#7dffb4',
                                  fontSize:
                                    9,
                                  fontWeight:
                                    800,
                                }}
                              >
                                {road}
                              </span>
                            ),
                          )}
                        </div>
                      ) : (
                        <strong
                          style={{
                            color:
                              '#7d8e96',
                            fontSize: 11,
                          }}
                        >
                          NONE
                        </strong>
                      )}
                    </div>
                  </div>

                  <div
                    style={{
                      padding: 14,
                      borderRadius: 10,
                      border:
                        '1px solid rgba(125,249,255,.1)',
                      background:
                        'rgba(125,249,255,.025)',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        gap: 8,
                        alignItems:
                          'flex-start',
                      }}
                    >
                      <BrainCircuit
                        size={15}
                        color="#7df9ff"
                      />

                      <div>
                        <span
                          style={{
                            display:
                              'block',
                            color:
                              '#7df9ff',
                            fontSize: 9,
                            fontWeight:
                              800,
                            letterSpacing:
                              '.09em',
                            marginBottom:
                              5,
                          }}
                        >
                          ENGINE EXPLANATION
                        </span>

                        <p
                          style={{
                            margin: 0,
                            color:
                              '#a8c0c9',
                            fontSize: 11,
                            lineHeight:
                              1.55,
                          }}
                        >
                          {
                            decision?.explanation
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* SIMULATION ROUTE IMPACTS                                           */}
      {/* ------------------------------------------------------------------ */}

      {simulationResult &&
        simulationResult.od_impacts.length >
          0 && (
          <section className="dashboard-section">
            <div className="section-header">
              <div>
                <div className="section-kicker">
                  COUNTERFACTUAL ROUTING
                </div>

                <h2>
                  DEMAND REDISTRIBUTION
                </h2>
              </div>

              <div className="section-badge">
                <Route size={14} />
                OD IMPACTS
              </div>
            </div>

            <div className="dashboard-panel">
              <div className="table-list">
                {simulationResult.od_impacts
                  .slice(0, 8)
                  .map((impact) => (
                    <div
                      className="table-row"
                      key={`${impact.origin}-${impact.destination}`}
                    >
                      <div className="route-cell">
                        <strong>
                          {impact.origin}
                          <ChevronRight
                            size={13}
                          />
                          {
                            impact.destination
                          }

                          {impact.route_changed && (
                            <span
                              style={{
                                marginLeft: 8,
                                padding:
                                  '3px 6px',
                                borderRadius:
                                  4,
                                background:
                                  'rgba(125,249,255,.08)',
                                color:
                                  '#7df9ff',
                                fontSize: 8,
                              }}
                            >
                              REROUTED
                            </span>
                          )}
                        </strong>

                        <small>
                          {impact.status} ·
                          demand{' '}
                          {formatNumber(
                            impact.demand,
                            2,
                          )}
                        </small>
                      </div>

                      <div
                        style={{
                          display:
                            'flex',
                          alignItems:
                            'center',
                          gap: 6,
                          color:
                            impact.route_changed
                              ? '#ffd166'
                              : '#708995',
                          fontSize: 10,
                        }}
                      >
                        <Clock3
                          size={13}
                        />

                        {impact.travel_time_delta_minutes !=
                        null
                          ? `${formatDelta(
                              impact.travel_time_delta_minutes,
                              3,
                            )} min`
                          : '—'}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </section>
        )}

      {/* ------------------------------------------------------------------ */}
      {/* PIPELINE                                                           */}
      {/* ------------------------------------------------------------------ */}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">
              INTELLIGENCE PIPELINE
            </div>

            <h2>
              OBSERVATION → DECISION
            </h2>
          </div>

          <div className="section-badge">
            <Activity size={14} />
            OPERATIONAL
          </div>
        </div>

        <div className="pipeline">
          {[
            {
              icon: Car,
              title: 'PERCEPTION',
              text: 'Vehicle detection, plate OCR and visual attributes.',
            },
            {
              icon: GitBranch,
              title: 'IDENTITY FUSION',
              text: 'Plate, appearance, time and spatial feasibility.',
            },
            {
              icon: Route,
              title: 'TRAJECTORY',
              text: 'Probabilistic route reconstruction under missing observations.',
            },
            {
              icon: Network,
              title: 'MOBILITY GRAPH',
              text: 'Aggregate inferred journeys into network demand.',
            },
            {
              icon: TriangleAlert,
              title: 'ANOMALIES',
              text: 'Detect coordinated changes across independent mobility signals.',
            },
            {
              icon: Zap,
              title: 'SIMULATION',
              text: 'Test interventions before changing the real network.',
            },
          ].map(
            ({
              icon: Icon,
              title,
              text,
            }) => (
              <div
                className="pipeline-node"
                key={title}
              >
                <div className="pipeline-icon">
                  <Icon size={17} />
                </div>

                <strong>
                  {title}
                </strong>

                <p>{text}</p>
              </div>
            ),
          )}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* FOOTER                                                             */}
      {/* ------------------------------------------------------------------ */}

      <footer className="footer">
        <div>
          URBANTRACKAI · CITY-WIDE AI
          MOBILITY ENGINE
        </div>

        <div>
          SYNTHETIC DEMONSTRATION DATA ·
          DECISION INTELLIGENCE
        </div>
      </footer>
    </div>
  )
}

export default App