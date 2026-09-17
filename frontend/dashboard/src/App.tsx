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
  XCircle,
  LockKeyhole,
  ChevronRight,
  Clock3,
  GitBranch,
  Gauge,
  Map as MapIcon,
  Maximize2,
  Network,
  Play,
  Route,
  ShieldCheck,
  TriangleAlert,
  Zap,
} from 'lucide-react'
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

async function postJson<T>(path: string, payload: unknown): Promise<T> {
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
    } catch {}
    throw new Error(message)
  }
  return response.json()
}

function formatNumber(value: number, digits = 1) {
  if (!Number.isFinite(value)) {
    return '—'
  }
  return value.toFixed(digits)
}

function formatPercent(value: number, digits = 1) {
  if (!Number.isFinite(value)) {
    return '—'
  }
  return `${(value * 100).toFixed(digits)}%`
}

function formatDelta(value: number, digits = 3) {
  if (!Number.isFinite(value)) {
    return '—'
  }
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function MobilityChart({
  traffic,
}: {
  traffic: TrafficResponse | null
}) {
  const summary = useMemo(() => {
    if (!traffic?.metrics.length) {
      return null
    }

    const metrics = traffic.metrics
    const totalFlow = metrics.reduce((sum, metric) => sum + (metric.hourly_flow ?? 0), 0)
    const averageUtilization = metrics.reduce((sum, metric) => sum + metric.utilization_ratio, 0) / metrics.length
    const averageCongestion = metrics.reduce((sum, metric) => sum + metric.congestion_score, 0) / metrics.length

    return {
      totalFlow,
      averageUtilization: averageUtilization * 100,
      averageCongestion,
      period: traffic.time_window_start && traffic.time_window_end
        ? `${traffic.time_window_start} - ${traffic.time_window_end}`
        : 'Current aggregation period',
    }
  }, [traffic])

  if (!summary) {
    return (
      <div className="chart-empty-state">
        No traffic metrics are available for the current aggregation period.
      </div>
    )
  }

  return (
    <div className="mobility-chart-card">
      <div className="chart-header">
        <div className="chart-legend">
          <span><i className="legend-dot flow-dot" /> Total flow</span>
          <span><i className="legend-dot utilization-dot" /> Utilization</span>
          <span><i className="legend-dot congestion-dot" /> Congestion score</span>
        </div>
        <div className="chart-window-label">CURRENT AGGREGATION PERIOD · {summary.period}</div>
      </div>

      <div className="mobility-summary-chart" role="img" aria-label="Current-window mobility performance">
        <div className="summary-chart-axis"><span>0</span><span>Active period</span></div>
        <div className="summary-bars">
          <div className="summary-bar-row">
            <span>Total flow</span>
            <div className="summary-bar-track"><i className="summary-bar flow-bar" style={{ width: `${Math.min(summary.totalFlow / Math.max(summary.totalFlow, 1) * 100, 100)}%` }} /></div>
            <strong>{formatNumber(summary.totalFlow)} veh/h</strong>
          </div>
          <div className="summary-bar-row">
            <span>Utilization</span>
            <div className="summary-bar-track"><i className="summary-bar utilization-bar" style={{ width: `${Math.min(summary.averageUtilization, 100)}%` }} /></div>
            <strong>{formatNumber(summary.averageUtilization)}%</strong>
          </div>
          <div className="summary-bar-row">
            <span>Congestion</span>
            <div className="summary-bar-track"><i className="summary-bar congestion-bar" style={{ width: `${Math.min(summary.averageCongestion, 100)}%` }} /></div>
            <strong>{formatNumber(summary.averageCongestion)}</strong>
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [network, setNetwork] = useState<NetworkResponse | null>(null)
  const [traffic, setTraffic] = useState<TrafficResponse | null>(null)
  const [od, setOd] = useState<OdResponse | null>(null)
  const [bottlenecks, setBottlenecks] = useState<BottleneckResponse | null>(null)
  const [anomalies, setAnomalies] = useState<AnomalyResponse | null>(null)
  const [trajectories, setTrajectories] = useState<Trajectory[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [mapFullscreen, setMapFullscreen] = useState(false)
  useEffect(() => {
    const locked = mapFullscreen
    document.body.style.overflow = locked ? 'hidden' : ''

    if (mapFullscreen) {
      const timer = window.setTimeout(() => {
        window.dispatchEvent(new Event('resize'))
      }, 120)

      return () => {
        window.clearTimeout(timer)
        document.body.style.overflow = ''
      }
    }

    return () => {
      document.body.style.overflow = ''
    }
  }, [mapFullscreen])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }
      if (mapFullscreen) {
        setMapFullscreen(false)
        setMapInteractive(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [mapFullscreen])

  const [simulationMode, setSimulationMode] = useState<SimulationMode>('closure')
  const [simulationRoadId, setSimulationRoadId] = useState('')
  const [simulationValue, setSimulationValue] = useState('')
  const [simulationLoading, setSimulationLoading] = useState(false)
  const [simulationError, setSimulationError] = useState<string | null>(null)
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null)
  const [mapInteractive, setMapInteractive] = useState(false)
  const [activeSection, setActiveSection] = useState('Dashboard')

  useEffect(() => {
    const sectionMap: Record<string, string> = {
      dashboard: 'Dashboard',
      'live-map': 'Live Map',
      analytics: 'Analytics',
      simulation: 'Simulation',
    }

    const sections = Object.keys(sectionMap)
      .map((id) => document.getElementById(id))
      .filter((section): section is HTMLElement => section !== null)

    if (!sections.length) {
      return
    }

    const updateActiveSection = () => {
      const activeLine = window.innerHeight * 0.35
      const currentSection = sections
        .map((section) => {
          const rect = section.getBoundingClientRect()
          const containsActiveLine = rect.top <= activeLine && rect.bottom > activeLine
          const distance = containsActiveLine
            ? 0
            : Math.abs(rect.top - activeLine)

          return { section, distance, containsActiveLine }
        })
        .sort(
          (a, b) =>
            Number(b.containsActiveLine) - Number(a.containsActiveLine) ||
            a.distance - b.distance
        )[0]

      if (currentSection) {
        setActiveSection((previous) => {
          const next = sectionMap[currentSection.section.id] ?? 'Dashboard'
          return previous === next ? previous : next
        })
      }
    }

    const observer = new IntersectionObserver(
      updateActiveSection,
      {
        root: null,
        threshold: [0, 0.25, 0.5, 0.75, 1],
      }
    )

    sections.forEach((section) => observer.observe(section))
    window.addEventListener('scroll', updateActiveSection, { passive: true })
    updateActiveSection()

    return () => {
      observer.disconnect()
      window.removeEventListener('scroll', updateActiveSection)
    }
  }, [loading])

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
          fetchJson<HealthResponse>('/health'),
          fetchJson<NetworkResponse>('/network'),
          fetchJson<TrafficResponse>('/traffic'),
          fetchJson<OdResponse>('/analytics/od'),
          fetchJson<BottleneckResponse>('/analytics/bottlenecks'),
          fetchJson<AnomalyResponse>('/anomalies'),
          fetchJson<TrajectoryResponse>('/trajectories'),
        ])

        if (cancelled) {
          return
        }

        setHealth(healthData)
        setNetwork(networkData)
        setTraffic(trafficData)
        setOd(odData)
        setBottlenecks(bottleneckData)
        setAnomalies(anomalyData)
        setTrajectories(trajectoryData.trajectories)

        if (!simulationRoadId && networkData.roads.length) {
          setSimulationRoadId(networkData.roads[0].road_id)
        }
      } catch (err) {
        if (cancelled) {
          return
        }
        setError(err instanceof Error ? err.message : 'Failed to load dashboard')
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
      traffic.metrics.reduce((sum, metric) => sum + metric.utilization_ratio, 0) /
      traffic.metrics.length

    return {
      averageUtilization,
      severeRoads: traffic.metrics.filter((metric) => metric.congestion_level === 'SEVERE').length,
      heavyRoads: traffic.metrics.filter((metric) => metric.congestion_level === 'HEAVY').length,
    }
  }, [traffic])

  const vehicleSearchResults = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) {
      return []
    }

    return trajectories
      .filter((trajectory) => trajectory.track_id.toLowerCase().includes(query))
      .slice(0, 6)
  }, [searchQuery, trajectories])

  const topOdPairs = useMemo(
    () =>
      [...(od?.pairs ?? [])]
        .sort((a, b) => b.demand - a.demand)
        .slice(0, 5),
    [od]
  )

  const significantRoadChanges = useMemo(
    () =>
      [...(anomalies?.road_evidence ?? [])]
        .filter((item) => typeof item.relative_change === 'number')
        .sort(
          (a, b) => Math.abs(b.relative_change ?? 0) - Math.abs(a.relative_change ?? 0)
        )
        .slice(0, 5),
    [anomalies]
  )

  const runSimulation = async () => {
    if (!simulationRoadId) {
      setSimulationError('Select a road before running the simulation.')
      return
    }

    const numericValue = Number(simulationValue)

    if (
      (simulationMode === 'capacity' || simulationMode === 'speed') &&
      (!Number.isFinite(numericValue) || numericValue <= 0)
    ) {
      setSimulationError('Enter a positive value for the selected intervention.')
      return
    }

    setSimulationLoading(true)
    setSimulationError(null)

    try {
      const scenarioId = `ui-${Date.now()}`

      const payload = {
        scenario_id: scenarioId,
        name:
          simulationMode === 'closure'
            ? `Close ${simulationRoadId}`
            : simulationMode === 'capacity'
            ? `Capacity reduction on ${simulationRoadId}`
            : `Speed reduction on ${simulationRoadId}`,
        description:
          simulationMode === 'closure'
            ? `Counterfactual closure of ${simulationRoadId}`
            : simulationMode === 'capacity'
            ? `Counterfactual capacity intervention on ${simulationRoadId}`
            : `Counterfactual speed intervention on ${simulationRoadId}`,
        closed_road_ids: simulationMode === 'closure' ? [simulationRoadId] : [],
        capacity_modifications_vph:
          simulationMode === 'capacity' ? { [simulationRoadId]: numericValue } : {},
        speed_modifications_kmph:
          simulationMode === 'speed' ? { [simulationRoadId]: numericValue } : {},
      }

      const result = await postJson<SimulationResult>('/simulation', payload)
      setSimulationResult(result)
    } catch (err) {
      setSimulationError(err instanceof Error ? err.message : 'Simulation failed')
      setSimulationResult(null)
    } finally {
      setSimulationLoading(false)
    }
  }

  const selectedSimulationRoad = network?.roads.find((road) => road.road_id === simulationRoadId)

  const sectionTargets: Record<string, string | undefined> = {
    Dashboard: '#dashboard',
    'Live Map': '#live-map',
    Analytics: '#analytics',
    Simulation: '#simulation',
  }

  const navItems = [
    { label: 'Dashboard', enabled: true },
    { label: 'Live Map', enabled: true },
    { label: 'Analytics', enabled: true },
    { label: 'Simulation', enabled: true },
  ] as const

  const navigateToSection = (target?: string, item?: string) => {
    if (!target) {
      return
    }

    if (item) {
      setActiveSection(item)
    }

    window.history.replaceState(null, '', target)
    document.querySelector(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  if (loading) {
    return (
      <div className="app-shell loading-shell">
        <div className="loading-card">
          <Activity size={20} />
          <span>INITIALIZING URBANTRACKAI</span>
        </div>
      </div>
    )
  }

  if (error || !network || !traffic) {
    return (
      <div className="app-shell loading-shell">
        <div className="loading-card error-card">
          <TriangleAlert size={20} />
          <span>{error ?? 'Dashboard data unavailable'}</span>
        </div>
      </div>
    )
  }

  const decision = simulationResult?.decision
  const recommendation = decision?.recommendation_class

  return (
    <div className="app-shell">
      <div className="dashboard-shell">
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="brand-mark">
              <Network size={20} />
            </div>
            <div>
              <div className="brand-name">URBANTRACKAI</div>
              <div className="brand-subtitle">SMARTER CITIES</div>
            </div>
          </div>

          <nav className="nav-list" aria-label="Main navigation">
            {navItems.map((item) => {
              const target = sectionTargets[item.label]
              const disabled = !item.enabled

              return (
                <button
                  key={item.label}
                  type="button"
                  className={`nav-item ${activeSection === item.label ? 'active' : ''}`}
                  disabled={disabled}
                  aria-disabled={disabled}
                  title={disabled ? `${item.label} is not available` : undefined}
                  onClick={() => !disabled && navigateToSection(target, item.label)}
                >
                  {item.label}
                </button>
              )
            })}
          </nav>
        </aside>

        <main className="dashboard">
          <header className="topbar">
            <div className="topbar-title">
              <div className="brand-mark small-mark">
                <Network size={17} />
              </div>
              <div className="brand-name small-name">URBANTRACKAI</div>
            </div>

            <div className="search-wrap">
              <input
                type="search"
                aria-label="Search vehicles"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search vehicles, roads, cameras..."
              />

              {vehicleSearchResults.length > 0 && (
                <div className="search-results">
                  {vehicleSearchResults.map((trajectory) => (
                    <button
                      key={trajectory.track_id}
                      type="button"
                      className="search-result"
                      onClick={() => {
                        setSelectedVehicleId(trajectory.track_id)
                        setSearchQuery(trajectory.track_id)
                        window.requestAnimationFrame(() => {
                          window.history.replaceState(null, '', '#live-map')
                          document.querySelector('#live-map')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                        })
                      }}
                    >
                      <div>
                        <strong>{trajectory.track_id}</strong>
                        <span>
                          {trajectory.origin_node} → {trajectory.destination_node}
                        </span>
                      </div>
                      <span className="search-action">VIEW ON MAP</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="topbar-status">
              <div className="status-pill">
                <div className="status-dot" />
                {health?.status === 'ok' ? 'ENGINE OPERATIONAL' : 'ENGINE DEGRADED'}
              </div>
            </div>
          </header>

          <section className="hero-section">
            <div>
              <div className="eyebrow">URBAN MOBILITY COMMAND CENTER</div>
              <h1>
                CITY TRAFFIC
                <br />
                <span>INTELLIGENCE</span>
              </h1>
              <p className="hero-copy">
                From distributed vehicle observations to probabilistic journeys, network-level anomalies and counterfactual decisions.
              </p>
            </div>

            <div className="hero-metrics">
              <div className="hero-metric">
                <span>TRAJECTORIES</span>
                <strong>{trajectories.length}</strong>
              </div>
              <div className="hero-metric">
                <span>TOTAL DEMAND</span>
                <strong>{formatNumber(od?.total_demand ?? 0)}</strong>
              </div>
              <div className="hero-metric">
                <span>AVG UTILIZATION</span>
                <strong>{formatPercent(trafficSummary.averageUtilization)}</strong>
              </div>
            </div>
          </section>

          <section className="metrics-grid">
            <div className="metric-card">
              <div className="metric-icon">
                <Car size={18} />
              </div>
              <div>
                <span>ACTIVE VEHICLES</span>
                <strong>{trajectories.length}</strong>
                <small>inferred trajectories</small>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-icon">
                <Gauge size={18} />
              </div>
              <div>
                <span>NETWORK UTILIZATION</span>
                <strong>{formatPercent(trafficSummary.averageUtilization)}</strong>
                <small>flow-weighted network state</small>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-icon">
                <AlertTriangle size={18} />
              </div>
              <div>
                <span>HEAVY / SEVERE ROADS</span>
                <strong>{trafficSummary.heavyRoads + trafficSummary.severeRoads}</strong>
                <small>congestion pressure points</small>
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-icon">
                <Route size={18} />
              </div>
              <div>
                <span>OD DEMAND</span>
                <strong>{formatNumber(od?.total_demand ?? 0)}</strong>
                <small>origin-destination trips</small>
              </div>
            </div>
          </section>

          <section id="dashboard" className="dashboard-section dashboard-anchor">
            <div className="section-header">
              <div>
                <div className="section-kicker">CITY MOBILITY GRAPH</div>
                <h2>Mobility performance</h2>
              </div>
              <div className="visual-header-actions">
                <div className="section-badge">
                  <Activity size={14} />
                  LIVE AGGREGATION
                </div>
              </div>
            </div>

            <div className="mobility-graph-panel">
              <MobilityChart traffic={traffic} />
            </div>
          </section>

          <section id="live-map" className="dashboard-section dashboard-anchor">
            <div className="section-header">
              <div>
                <div className="section-kicker">GEOSPATIAL INTELLIGENCE</div>
                <h2>LIVE MOBILITY MAP</h2>
              </div>
              <div className="visual-header-actions">
                <div className="section-badge">
                  <MapIcon size={14} />
                  GIS LAYER
                </div>
                <button
                  type="button"
                  className="interaction-button"
                  onClick={() => setMapInteractive((current) => !current)}
                >
                  <LockKeyhole size={14} />
                  {mapInteractive ? 'LOCK MAP' : 'UNLOCK MAP'}
                </button>
                <button
                  type="button"
                  className="interaction-button"
                  onClick={() => {
                    setMapInteractive(true)
                    setMapFullscreen(true)
                  }}
                >
                  <Maximize2 size={14} />
                  FULLSCREEN MAP
                </button>
              </div>
            </div>

            <CityMap
              network={network}
              traffic={traffic.metrics}
              trajectories={trajectories}
              interactive={mapInteractive}
              fullscreen={mapFullscreen}
              selectedTrackId={selectedVehicleId}
              onSelectTrack={setSelectedVehicleId}
              onExitInteraction={() => {
                setMapInteractive(false)
                setMapFullscreen(false)
              }}
            />
          </section>

      <section id="analytics" className="dashboard-section dashboard-anchor">
        <div className="section-header">
          <div>
            <div className="section-kicker">NETWORK ANOMALY INTELLIGENCE</div>
            <h2>MOBILITY STATE MONITOR</h2>
          </div>
          <div className="section-badge">
            <BrainCircuit size={14} />
            MULTI-SIGNAL DETECTOR
          </div>
        </div>

        <div className="anomaly-grid">
          <div className="anomaly-main">
            <div
              className={`anomaly-status-row ${
                anomalies?.anomaly_events?.length ? 'warning' : 'normal'
              }`}
            >
              <div
                className={`anomaly-status ${
                  anomalies?.anomaly_events?.length ? 'warning' : 'normal'
                }`}
              >
                {anomalies?.anomaly_events?.length ? (
                  <TriangleAlert size={18} />
                ) : (
                  <ShieldCheck size={18} />
                )}
                <div>
                  <span>DETECTOR ASSESSMENT</span>
                  <strong>
                    {anomalies?.anomaly_events?.length ? 'ANOMALY SIGNAL DETECTED' : 'NO CONSENSUS ANOMALY'}
                  </strong>
                </div>
              </div>

              <div className="evidence-count">
                <span>EVIDENCE SIGNALS</span>
                <strong>{anomalies?.anomaly_events?.length ?? 0}</strong>
              </div>
            </div>

            <div className="anomaly-stats">
              <div>
                <span>BASELINE STATE</span>
                <strong>{anomalies?.baseline_snapshot_id ?? '—'}</strong>
              </div>
              <div>
                <span>CURRENT STATE</span>
                <strong>{anomalies?.current_snapshot_id ?? '—'}</strong>
              </div>
              <div>
                <span>OD DIVERGENCE</span>
                <strong>{formatNumber(anomalies?.od_divergence ?? 0, 4)}</strong>
                <small>Jensen–Shannon divergence</small>
              </div>
              <div>
                <span>ROUTE DIVERGENCE</span>
                <strong>{formatNumber(anomalies?.route_divergence ?? 0, 4)}</strong>
                <small>route redistribution</small>
              </div>
              <div>
                <span>HHI DELTA</span>
                <strong>{formatDelta(anomalies?.hhi_delta ?? 0, 4)}</strong>
                <small>network concentration</small>
              </div>
              <div>
                <span>BOTTLENECK SHIFT</span>
                <strong>{anomalies?.bottleneck_changes?.length ?? 0}</strong>
                <small>detected changes</small>
              </div>
            </div>

            <div className="anomaly-explanation">
              <span>DETECTOR ASSESSMENT</span>
              <p>
                {anomalies?.anomaly_events?.length
                  ? 'Multiple mobility signals have crossed configured detection thresholds and require investigation.'
                  : 'Current mobility deviations remain below configured detection thresholds.'}
              </p>
              <small>
                The detector compares the current mobility state against a synthetic normal baseline using flow, OD demand, route redistribution, network concentration and bottleneck evidence.
              </small>
            </div>
          </div>

          <div className="anomaly-side">
            <div className="side-panel-title">SIGNIFICANT ROAD CHANGES</div>
            <div className="change-list">
              {significantRoadChanges.map((item) => {
                const change = item.relative_change ?? 0
                const positive = change >= 0
                return (
                  <div className="change-row" key={item.road_id}>
                    <span>{item.road_id}</span>
                    <strong className={positive ? 'positive' : 'negative'}>
                      {positive ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                      {Math.abs(change * 100).toFixed(1)}%
                    </strong>
                  </div>
                )
              })}
              {!significantRoadChanges.length && (
                <div className="empty-state">No material road changes.</div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="two-column-grid">
        <div className="dashboard-panel">
          <div className="panel-heading">
            <div>
              <span>ORIGIN / DESTINATION</span>
              <h3>TOP MOBILITY PAIRS</h3>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="table-list">
            {topOdPairs.map((pair, index) => (
              <div className="table-row" key={`${pair.origin}-${pair.destination}`}>
                <span className="rank">{String(index + 1).padStart(2, '0')}</span>
                <div className="route-cell">
                  <strong>
                    {pair.origin}
                    <ChevronRight size={13} />
                    {pair.destination}
                  </strong>
                  <small>demand flow</small>
                </div>
                <b>{formatNumber(pair.demand)}</b>
              </div>
            ))}
          </div>
        </div>

        <div className="dashboard-panel">
          <div className="panel-heading">
            <div>
              <span>TRAFFIC PRESSURE</span>
              <h3>BOTTLENECK INTELLIGENCE</h3>
            </div>
            <Gauge size={18} />
          </div>
          <div className="table-list">
            {bottlenecks?.bottlenecks?.slice(0, 5).map((item) => (
              <div className="table-row" key={item.road_id}>
                <div className="route-cell">
                  <strong>{item.road_id}</strong>
                  <small>{item.severity}</small>
                </div>
                <div className="mini-bar">
                  <span
                    style={{
                      width: `${Math.min(item.utilization_ratio * 100, 100)}%`,
                    }}
                  />
                </div>
                <b>{formatPercent(item.utilization_ratio)}</b>
              </div>
            ))}
            {!bottlenecks?.bottlenecks?.length && (
              <div className="empty-state">No bottlenecks detected.</div>
            )}
          </div>
        </div>
      </section>

      <section id="simulation" className="dashboard-section dashboard-anchor">
        <div className="section-header">
          <div>
            <div className="section-kicker">DECISION INTELLIGENCE</div>
            <h2>What-if simulation</h2>
          </div>
          <div className="section-badge simulation-badge">
            <Zap size={14} />
            COUNTERFACTUAL ENGINE
          </div>
        </div>

        <div className="simulation-shell">
          <div className="simulation-controls">
            <div className="simulation-intro">
              <span>COUNTERFACTUAL SCENARIO</span>
              <h3>Test a network intervention</h3>
              <p>Estimate rerouting, travel-time impact and network consequences using the live mobility model.</p>
            </div>

            <div className="simulation-form">
              <label>
                <span>INTERVENTION</span>
                <select
                  value={simulationMode}
                  onChange={(event) => {
                    setSimulationMode(event.target.value as SimulationMode)
                    setSimulationResult(null)
                    setSimulationError(null)
                  }}
                >
                  <option value="closure">Close road</option>
                  <option value="capacity">Reduce capacity</option>
                  <option value="speed">Reduce speed limit</option>
                </select>
              </label>

              <label>
                <span>TARGET ROAD</span>
                <select
                  value={simulationRoadId}
                  onChange={(event) => {
                    setSimulationRoadId(event.target.value)
                    setSimulationResult(null)
                    setSimulationError(null)
                  }}
                >
                  {network.roads.map((road) => (
                    <option key={road.road_id} value={road.road_id}>
                      {road.road_id} · {road.from_node} → {road.to_node}
                    </option>
                  ))}
                </select>
              </label>

              {simulationMode !== 'closure' && (
                <label>
                  <span>{simulationMode === 'capacity' ? 'NEW CAPACITY · VPH' : 'NEW SPEED LIMIT · KM/H'}</span>
                  <input
                    type="number"
                    min="0.01"
                    step="1"
                    value={simulationValue}
                    onChange={(event) => {
                      setSimulationValue(event.target.value)
                      setSimulationError(null)
                    }}
                    placeholder={simulationMode === 'capacity' ? 'e.g. 800' : 'e.g. 25'}
                  />
                </label>
              )}

              {selectedSimulationRoad && (
                <div className="simulation-road-facts">
                  <div><span>CAPACITY</span><strong>{selectedSimulationRoad.capacity_vph.toLocaleString()} <small>vph</small></strong></div>
                  <div><span>SPEED LIMIT</span><strong>{selectedSimulationRoad.speed_limit_kmph} <small>km/h</small></strong></div>
                </div>
              )}

              {simulationError && (
                <div className="simulation-error"><TriangleAlert size={15} /><span>{simulationError}</span></div>
              )}

              <button type="button" className="simulation-run-button" onClick={runSimulation} disabled={simulationLoading}>
                <Play size={15} />
                {simulationLoading ? 'RUNNING SIMULATION...' : 'RUN SIMULATION'}
              </button>
            </div>
          </div>

          <div className="simulation-result">
            {!simulationResult ? (
              <div className="simulation-empty">
                <GitBranch size={25} />
                <strong>Simulation ready</strong>
                <p>Choose an intervention and run the counterfactual engine to evaluate how demand redistributes through the network.</p>
              </div>
            ) : (
              <>
                <div className="simulation-result-heading">
                  <div>
                    <span>SIMULATION RESULT</span>
                    <h3>{simulationResult.scenario.name}</h3>
                  </div>
                  <strong className={`recommendation-badge ${recommendation?.toLowerCase() ?? 'neutral'}`}>
                    {recommendation === 'FAVORABLE' ? <CheckCircle2 size={14} /> : recommendation === 'UNFAVORABLE' ? <XCircle size={14} /> : <TriangleAlert size={14} />}
                    {recommendation ?? 'UNKNOWN'}
                  </strong>
                </div>

                <div className="simulation-metrics">
                  <div><span>REROUTED DEMAND</span><strong>{formatNumber(decision?.rerouted_demand ?? 0, 2)}</strong></div>
                  <div><span>UNROUTABLE DEMAND</span><strong className={decision?.unroutable_demand ? 'danger-text' : 'success-text'}>{formatNumber(decision?.unroutable_demand ?? 0, 2)}</strong></div>
                  <div><span>TRAVEL-TIME CHANGE</span><strong>{formatDelta(decision?.total_travel_time_delta_minutes ?? 0, 3)} <small>min</small></strong></div>
                  <div><span>CONGESTION CHANGE</span><strong>{formatDelta(decision?.network_congestion_delta ?? 0, 4)}</strong></div>
                </div>

                <div className="simulation-impact-grid">
                  <div><span>NEWLY AFFECTED ROADS</span>{decision?.newly_congested_roads?.length ? <div className="road-tags danger-tags">{decision.newly_congested_roads.map((road) => <b key={road}>{road}</b>)}</div> : <strong className="success-text">NONE</strong>}</div>
                  <div><span>RELIEVED ROADS</span>{decision?.relieved_roads?.length ? <div className="road-tags success-tags">{decision.relieved_roads.map((road) => <b key={road}>{road}</b>)}</div> : <strong>NONE</strong>}</div>
                </div>

                <div className="simulation-explanation"><BrainCircuit size={16} /><div><span>ENGINE EXPLANATION</span><p>{decision?.explanation}</p></div></div>
              </>
            )}
          </div>
        </div>
      </section>

      {simulationResult && simulationResult.od_impacts.length > 0 && (
        <section className="dashboard-section">
          <div className="section-header">
            <div><div className="section-kicker">COUNTERFACTUAL ROUTING</div><h2>Demand redistribution</h2></div>
            <div className="section-badge"><Route size={14} /> OD IMPACTS</div>
          </div>
          <div className="dashboard-panel">
            <div className="table-list">
              {simulationResult.od_impacts.slice(0, 8).map((impact, index) => (
                <div className="table-row" key={`${impact.origin}-${impact.destination}-${index}`}>
                  <div className="route-cell"><strong>{impact.origin}<ChevronRight size={13} />{impact.destination}{impact.route_changed && <em className="route-change-tag">REROUTED</em>}</strong><small>{impact.status} · demand {formatNumber(impact.demand, 2)}</small></div>
                  <div className="impact-time"><Clock3 size={13} />{impact.travel_time_delta_minutes != null ? `${formatDelta(impact.travel_time_delta_minutes, 3)} min` : '—'}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <div className="section-kicker">INTELLIGENCE PIPELINE</div>
            <h2>OBSERVATION → DECISION</h2>
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
              icon: Network,
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
              icon: Activity,
              title: 'SIMULATION',
              text: 'Test interventions before changing the real network.',
            },
          ].map(({ icon: Icon, title, text }) => (
            <div className="pipeline-node" key={title}>
              <div className="pipeline-icon">
                <Icon size={17} />
              </div>
              <strong>{title}</strong>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="footer">
        <div>URBANTRACKAI · CITY-WIDE AI MOBILITY ENGINE</div>
        <div>SYNTHETIC DEMONSTRATION DATA · DECISION INTELLIGENCE</div>
      </footer>
        </main>
      </div>
    </div>
  )
}

export default App