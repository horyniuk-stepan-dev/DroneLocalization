import csv
import json
from pathlib import Path
from datetime import datetime
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResultExporter:
    """Експорт результатів локалізації у різні формати."""

    @staticmethod
    def export_csv(results: list[dict], output_path: str):
        """
        Експорт у CSV файл.

        Args:
            results: список словників з ключами:
                frame_id, lat, lon, confidence, timestamp, matched_frame, inliers
            output_path: шлях до вихідного файлу
        """
        if not results:
            logger.warning("No results to export")
            return

        fieldnames = ['frame_id', 'timestamp', 'lat', 'lon', 'confidence', 'matched_frame', 'inliers']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.success(f"Exported {len(results)} results to CSV: {output_path}")

    @staticmethod
    def export_geojson(results: list[dict], output_path: str):
        """Експорт у GeoJSON (для GIS-систем)."""
        features = []
        for r in results:
            if 'lat' not in r or 'lon' not in r:
                continue
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r['lon'], r['lat']]
                },
                "properties": {
                    "frame_id": r.get('frame_id'),
                    "confidence": r.get('confidence'),
                    "timestamp": r.get('timestamp'),
                    "matched_frame": r.get('matched_frame'),
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "exported_at": datetime.now().isoformat(),
                "total_points": len(features),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        logger.success(f"Exported {len(features)} points to GeoJSON: {output_path}")

    @staticmethod
    def export_kml(results: list[dict], output_path: str, name: str = "Drone Track"):
        """Експорт у KML (для Google Earth)."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            f'  <name>{name}</name>',
            f'  <description>Exported {datetime.now().strftime("%Y-%m-%d %H:%M")}</description>',
        ]

        # Стиль маркера
        lines.extend([
            '  <Style id="dronePoint">',
            '    <IconStyle>',
            '      <scale>0.6</scale>',
            '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/airports.png</href></Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # Точки
        for r in results:
            if 'lat' not in r or 'lon' not in r:
                continue
            conf = r.get('confidence', 0)
            fid = r.get('frame_id', '?')
            lines.extend([
                '  <Placemark>',
                f'    <name>Frame {fid}</name>',
                f'    <description>Confidence: {conf:.2f}</description>',
                '    <styleUrl>#dronePoint</styleUrl>',
                '    <Point>',
                f'      <coordinates>{r["lon"]},{r["lat"]},0</coordinates>',
                '    </Point>',
                '  </Placemark>',
            ])

        # Трек (лінія)
        coords_str = " ".join(
            f"{r['lon']},{r['lat']},0"
            for r in results if 'lat' in r and 'lon' in r
        )
        if coords_str:
            lines.extend([
                '  <Placemark>',
                f'    <name>{name} - Path</name>',
                '    <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>',
                '    <LineString>',
                '      <tessellate>1</tessellate>',
                f'      <coordinates>{coords_str}</coordinates>',
                '    </LineString>',
                '  </Placemark>',
            ])

        lines.extend(['</Document>', '</kml>'])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.success(f"Exported {len(results)} points to KML: {output_path}")
