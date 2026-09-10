import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Tooltip,
  useMap,
} from 'react-leaflet'
import { useEffect, useMemo, useState } from 'react'
import 'leaflet/dist/leaflet.css'

import type {
  NetworkResponse,
  TrafficMetric,
  Trajectory,
} from '../types'

type CityMapProps = {
  network: NetworkResponse
  traffic?: TrafficMetric[]
  trajectories?: Trajectory[]
  interactive?: boolean
  fullscreen?: boolean
  onExitInteraction?: () => void
}

function MapController({
  network,
}: {
  network: NetworkResponse
}) {
  const map = useMap()

  useEffect(() => {
    const points = network.nodes
      .filter(
        (node) =>
          typeof node.lat === 'number' &&
          typeof node.lon === 'number',
      )
      .map(
        (node) =>
          [node.lat!, node.lon!] as [
            number,
            number,
          ],
      )

    if (!points.length) {
      return
    }

    const timer = window.setTimeout(() => {
      map.invalidateSize(true)
      map.fitBounds(points, {
        padding: [40, 40],
      })
    }, 250)

    return () => {
      window.clearTimeout(timer)
    }
  }, [map, network])

  return null
}

function roadColor(metric?: TrafficMetric) {
  if (!metric) {
    return '#31586b'
  }

  switch (metric.congestion_level) {
    case 'SEVERE':
      return '#ff3b5c'
    case 'HEAVY':
      return '#ff8a3d'
    case 'MODERATE':
      return '#ffd166'
    default:
      return '#36d1dc'
  }
}

function roadWeight(metric?: TrafficMetric) {
  if (!metric) {
    return 5
  }

  switch (metric.congestion_level) {
    case 'SEVERE':
      return 9
    case 'HEAVY':
      return 8
    case 'MODERATE':
      return 7
    default:
      return 5
  }
}

export default function CityMap({
  network,
  traffic = [],
  trajectories = [],
  interactive = false,
  fullscreen = false,
  onExitInteraction,
}: CityMapProps) {
  const [selectedRoad, setSelectedRoad] =
    useState<string | null>(null)

  const [selectedTrackId, setSelectedTrackId] =
    useState<string | null>(null)

  const nodes = useMemo(
    () =>
      network.nodes.filter(
        (node) =>
          typeof node.lat === 'number' &&
          typeof node.lon === 'number',
      ),
    [network],
  )

  const nodeLookup = useMemo(
    () =>
      new Map(
        network.nodes.map((node) => [
          node.node_id,
          node,
        ]),
      ),
    [network],
  )

  const trafficLookup = useMemo(
    () =>
      new Map(
        traffic.map((metric) => [
          metric.road_id,
          metric,
        ]),
      ),
    [traffic],
  )

  const selectedTrajectory = useMemo(
    () =>
      trajectories.find(
        (trajectory) =>
          trajectory.track_id ===
          selectedTrackId,
      ),
    [trajectories, selectedTrackId],
  )

  const trajectoryColors = [
    '#ffffff',
    '#ff4fd8',
    '#7df9ff',
    '#ffd166',
    '#a7ff83',
  ]

  if (!nodes.length) {
    return (
      <div className="map-empty">
        No geographic coordinates available.
      </div>
    )
  }

  const center: [number, number] = [
    nodes.reduce(
      (sum, node) => sum + node.lat!,
      0,
    ) / nodes.length,

    nodes.reduce(
      (sum, node) => sum + node.lon!,
      0,
    ) / nodes.length,
  ]

  const selectedRoadData =
    selectedRoad
      ? network.roads.find(
          (road) =>
            road.road_id === selectedRoad,
        )
      : undefined

  const selectedMetric =
    selectedRoad
      ? trafficLookup.get(selectedRoad)
      : undefined

  return (
    <div
      className={`city-map ${
        fullscreen ? 'city-map-fullscreen' : ''
      }`}
    >
      <MapContainer
        center={center}
        zoom={14}
        scrollWheelZoom={interactive}
        dragging={interactive}
        doubleClickZoom={interactive}
        touchZoom={interactive}
        keyboard={interactive}
        className="leaflet-map"
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapController
          network={network}
        />

        {/* Base road network */}
        {network.roads.map((road) => {
          const from =
            nodeLookup.get(road.from_node)

          const to =
            nodeLookup.get(road.to_node)

          if (
            !from ||
            !to ||
            typeof from.lat !== 'number' ||
            typeof from.lon !== 'number' ||
            typeof to.lat !== 'number' ||
            typeof to.lon !== 'number'
          ) {
            return null
          }

          const metric =
            trafficLookup.get(road.road_id)

          const selected =
            selectedRoad === road.road_id

          return (
            <Polyline
              key={road.road_id}
              positions={[
                [from.lat, from.lon],
                [to.lat, to.lon],
              ]}
              pathOptions={{
                color: road.is_closed
                  ? '#59636d'
                  : selected
                    ? '#ffffff'
                    : roadColor(metric),

                weight: selected
                  ? 11
                  : road.is_closed
                    ? 5
                    : roadWeight(metric),

                opacity: selected
                  ? 1
                  : road.is_closed
                    ? 0.55
                    : 0.9,

                dashArray:
                  road.is_closed
                    ? '8 8'
                    : undefined,
              }}
              eventHandlers={{
                click: () => {
                  setSelectedRoad(
                    road.road_id,
                  )
                },
              }}
            >
              <Tooltip sticky>
                <strong>
                  {road.road_id}
                </strong>

                <br />

                {road.from_node}
                {' → '}
                {road.to_node}

                <br />

                {road.is_closed ? (
                  'ROAD CLOSED'
                ) : metric ? (
                  <>
                    {metric.hourly_flow?.toFixed(
                      1,
                    ) ?? '—'}{' '}
                    veh/h
                    <br />
                    {metric.congestion_level}
                    <br />
                    {(
                      metric.utilization_ratio *
                      100
                    ).toFixed(1)}
                    % utilization
                  </>
                ) : (
                  'Traffic unavailable'
                )}
              </Tooltip>
            </Polyline>
          )
        })}

        {/* Probabilistic trajectory overlays */}
        {selectedTrajectory?.candidate_routes.map(
          (route, index) => {
            const routePositions = route.nodes
              .map((nodeId) => {
                const node =
                  nodeLookup.get(nodeId)

                if (
                  !node ||
                  typeof node.lat !==
                    'number' ||
                  typeof node.lon !==
                    'number'
                ) {
                  return null
                }

                return [
                  node.lat,
                  node.lon,
                ] as [number, number]
              })
              .filter(
                (
                  point,
                ): point is [
                  number,
                  number,
                ] => point !== null,
              )

            if (routePositions.length < 2) {
              return null
            }

            const probability =
              route.probability * 100

            return (
              <Polyline
                key={`trajectory-${selectedTrajectory.track_id}-${index}`}
                positions={routePositions}
                pathOptions={{
                  color:
                    trajectoryColors[
                      index %
                        trajectoryColors.length
                    ],
                  weight:
                    index === 0
                      ? 7
                      : 5,
                  opacity:
                    Math.max(
                      0.45,
                      Math.min(
                        1,
                        0.35 +
                          route.probability,
                      ),
                    ),
                  dashArray:
                    index === 0
                      ? undefined
                      : '10 8',
                }}
              >
                <Tooltip sticky>
                  <strong>
                    {selectedTrajectory.track_id}
                  </strong>

                  <br />

                  Candidate route {index + 1}

                  <br />

                  Probability:{' '}
                  {probability.toFixed(1)}%

                  <br />

                  {route.nodes.join(' → ')}
                </Tooltip>
              </Polyline>
            )
          },
        )}

        {nodes.map((node) => (
          <CircleMarker
            key={node.node_id}
            center={[
              node.lat!,
              node.lon!,
            ]}
            radius={7}
            pathOptions={{
              color: '#07141b',
              weight: 2,
              fillColor: '#39d8ff',
              fillOpacity: 1,
            }}
          >
            <Tooltip>
              <strong>
                {node.node_id}
              </strong>

              {node.name && (
                <>
                  <br />
                  {node.name}
                </>
              )}
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>

      {/* Interaction mode controls */}
      {fullscreen && (
        <div className="map-interaction-overlay">
          <div className="interaction-overlay-title">
            <div>
              <span>
                MAP INTERACTION MODE
              </span>

              <strong>
                Live Traffic Map
              </strong>
            </div>

            <button
              className="interaction-exit-button"
              onClick={
                onExitInteraction
              }
            >
              <span>×</span>
              EXIT FULL SCREEN
            </button>
          </div>

          <div className="interaction-help">
            <span>
              SCROLL TO ZOOM
            </span>

            <span>
              DRAG TO PAN
            </span>

            <span>
              CLICK ROADS FOR DETAILS
            </span>

            <span>
              ESC TO EXIT
            </span>
          </div>
        </div>
      )}

      {/* Trajectory selector */}
      {trajectories.length > 0 && (
        <div className="trajectory-control">
          <span className="trajectory-control-label">
            VEHICLE TRAJECTORY
          </span>

          <select
            value={selectedTrackId ?? ''}
            onChange={(event) =>
              setSelectedTrackId(
                event.target.value ||
                  null,
              )
            }
          >
            <option value="">
              SELECT VEHICLE
            </option>

            {trajectories.map(
              (trajectory) => (
                <option
                  key={trajectory.track_id}
                  value={
                    trajectory.track_id
                  }
                >
                  {trajectory.track_id}
                </option>
              ),
            )}
          </select>

          {selectedTrajectory && (
            <div className="trajectory-control-meta">
              {selectedTrajectory.origin_node}
              {' → '}
              {selectedTrajectory.destination_node}
            </div>
          )}
        </div>
      )}

      <div className="map-legend">
        <span>
          <i className="legend-free" />
          FREE
        </span>

        <span>
          <i className="legend-moderate" />
          MODERATE
        </span>

        <span>
          <i className="legend-heavy" />
          HEAVY
        </span>

        <span>
          <i className="legend-severe" />
          SEVERE
        </span>

        {selectedTrajectory && (
          <>
            <span>
              <i className="legend-trajectory" />
              TRAJECTORY
            </span>
          </>
        )}
      </div>

      {selectedTrajectory && (
        <div className="trajectory-panel">
          <div className="trajectory-panel-header">
            <div>
              <span>
                PROBABILISTIC TRAJECTORY
              </span>

              <strong>
                {selectedTrajectory.track_id}
              </strong>
            </div>

            <button
              className="road-close"
              onClick={() =>
                setSelectedTrackId(null)
              }
            >
              ×
            </button>
          </div>

          <div className="trajectory-route">
            <span>
              {selectedTrajectory.origin_node}
            </span>

            <strong>→</strong>

            <span>
              {selectedTrajectory.destination_node}
            </span>
          </div>

          <div className="trajectory-summary">
            <div>
              <span>
                CANDIDATE ROUTES
              </span>

              <strong>
                {
                  selectedTrajectory
                    .candidate_routes
                    .length
                }
              </strong>
            </div>

            <div>
              <span>
                VEHICLE WEIGHT
              </span>

              <strong>
                {selectedTrajectory.vehicle_weight.toFixed(
                  1,
                )}
              </strong>
            </div>
          </div>

          <div className="trajectory-routes">
            {selectedTrajectory.candidate_routes
              .map(
                (route, index) => (
                  <div
                    className="trajectory-route-row"
                    key={index}
                  >
                    <span
                      className="trajectory-route-dot"
                      style={{
                        background:
                          trajectoryColors[
                            index %
                              trajectoryColors.length
                          ],
                      }}
                    />

                    <div>
                      <strong>
                        ROUTE {index + 1}
                      </strong>

                      <small>
                        {route.nodes.join(
                          ' → ',
                        )}
                      </small>
                    </div>

                    <b>
                      {(
                        route.probability *
                        100
                      ).toFixed(1)}
                      %
                    </b>
                  </div>
                ),
              )}
          </div>

          <div className="trajectory-note">
            Probability represents the estimated
            relative likelihood of each candidate
            route from the upstream trajectory
            inference engine.
          </div>
        </div>
      )}

      {selectedRoadData &&
        !selectedTrajectory && (
          <div className="road-detail-panel">
            <div className="road-detail-header">
              <div>
                <span className="road-detail-label">
                  SELECTED ROAD
                </span>

                <h3>
                  {selectedRoadData.road_id}
                </h3>
              </div>

              <button
                className="road-close"
                onClick={() =>
                  setSelectedRoad(null)
                }
              >
                ×
              </button>
            </div>

            <div className="road-route">
              {selectedRoadData.from_node}
              <span>→</span>
              {selectedRoadData.to_node}
            </div>

            <div className="road-detail-grid">
              <div>
                <span>FLOW</span>

                <strong>
                  {selectedMetric?.hourly_flow?.toFixed(
                    1,
                  ) ?? '—'}
                </strong>

                <small>veh/h</small>
              </div>

              <div>
                <span>UTILIZATION</span>

                <strong>
                  {selectedMetric
                    ? `${(
                        selectedMetric.utilization_ratio *
                        100
                      ).toFixed(1)}%`
                    : '—'}
                </strong>
              </div>

              <div>
                <span>STATUS</span>

                <strong>
                  {selectedRoadData.is_closed
                    ? 'CLOSED'
                    : selectedMetric?.congestion_level ??
                      'UNKNOWN'}
                </strong>
              </div>

              <div>
                <span>CAPACITY</span>

                <strong>
                  {selectedRoadData.capacity_vph.toLocaleString()}
                </strong>

                <small>veh/h</small>
              </div>
            </div>
          </div>
        )}

      {!selectedRoad &&
        !selectedTrajectory && (
          <div className="map-hint">
            CLICK A ROAD SEGMENT TO INSPECT
          </div>
        )}
    </div>
  )
}