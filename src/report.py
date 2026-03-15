"""
Violation Report Generator

Generates structured violation reports from the SQLite database.
Supports CSV and HTML export formats with summary statistics.

Based on PPT:
  "Generate structured violation records with timestamp and visual evidence"
"""
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ReportGenerator:
    """
    Generates violation reports from the database.
    
    Supports:
    - CSV export (tabular data)
    - HTML report (styled, with embedded evidence images)
    - Summary statistics for dashboard display
    """
    
    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory for saving reports. Default: reports/ in project root
        """
        self.output_dir = output_dir or str(Path(__file__).parent.parent / "reports")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_csv(self, records: List, filename: str = None) -> str:
        """
        Export violation records to CSV.
        
        Args:
            records: List of ViolationRecord objects (from database.py)
            filename: Optional output filename
            
        Returns:
            Path to the generated CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations_report_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Timestamp', 'Violation Type', 'Track ID',
                'Plate Number', 'Confidence (%)', 'Evidence File', 'Details'
            ])
            for r in records:
                writer.writerow([
                    r.id,
                    r.timestamp,
                    r.violation_type.replace('_', ' ').title(),
                    r.track_id or '-',
                    r.plate_number or 'Not Detected',
                    f"{r.confidence * 100:.1f}",
                    os.path.basename(r.evidence_path) if r.evidence_path else '-',
                    r.details
                ])
        
        print(f"[Report] CSV exported: {filepath} ({len(records)} records)")
        return filepath
    
    def generate_html(self, records: List, stats: Dict[str, int],
                      title: str = "Traffic Violation Report",
                      filename: str = None) -> str:
        """
        Generate a styled HTML report with statistics and violation table.
        
        Args:
            records: List of ViolationRecord objects
            stats: Dict of violation_type -> count (from db.get_stats())
            title: Report title
            filename: Optional output filename
            
        Returns:
            Path to the generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations_report_{timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        total = sum(stats.values()) if stats else 0
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build stats cards HTML
        stats_html = ""
        colors = {
            'no_helmet': '#e74c3c',
            'triple_riding': '#e67e22',
            'phone_usage': '#9b59b6',
            'red_signal': '#c0392b',
            'missing_plate': '#2c3e50',
            'overspeed': '#d35400',
            'wrong_way': '#8e44ad',
        }
        for vtype, count in sorted(stats.items(), key=lambda x: -x[1]):
            color = colors.get(vtype, '#34495e')
            label = vtype.replace('_', ' ').title()
            stats_html += f"""
            <div class="stat-card" style="border-left: 4px solid {color};">
                <div class="stat-count">{count}</div>
                <div class="stat-label">{label}</div>
            </div>"""
        
        # Build table rows
        rows_html = ""
        for r in records:
            vtype_label = r.violation_type.replace('_', ' ').title()
            color = colors.get(r.violation_type, '#34495e')
            plate = r.plate_number or '<span class="na">Not Detected</span>'
            evidence = ""
            if r.evidence_path and os.path.exists(r.evidence_path):
                evidence = f'<a href="file:///{r.evidence_path}" target="_blank">View</a>'
            else:
                evidence = '<span class="na">-</span>'
            
            rows_html += f"""
            <tr>
                <td>{r.id}</td>
                <td>{r.timestamp}</td>
                <td><span class="badge" style="background:{color};">{vtype_label}</span></td>
                <td>{r.track_id or '-'}</td>
                <td>{plate}</td>
                <td>{r.confidence * 100:.1f}%</td>
                <td>{evidence}</td>
                <td class="details">{r.details}</td>
            </tr>"""
        
        # Full HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            padding: 30px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 25px;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 5px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stat-count {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 13px; color: #7f8c8d; margin-top: 4px; }}
        .total-bar {{
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 25px;
            font-size: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .total-bar strong {{ color: #e74c3c; font-size: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-size: 13px;
            text-transform: uppercase;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #ecf0f1;
            font-size: 13px;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        .na {{ color: #bdc3c7; }}
        .details {{ max-width: 200px; font-size: 12px; color: #7f8c8d; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #95a5a6;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="meta">Generated: {generated_at} | AI-Based Traffic Violation Detection System</div>
    </div>
    
    <div class="total-bar">
        Total Violations Recorded: <strong>{total}</strong>
    </div>
    
    <div class="stats-grid">
        {stats_html}
    </div>
    
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Timestamp</th>
                <th>Violation</th>
                <th>Track</th>
                <th>Plate</th>
                <th>Confidence</th>
                <th>Evidence</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    
    <div class="footer">
        Traffic Violation Detection System &mdash; AI &amp; ML Department
    </div>
</body>
</html>"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[Report] HTML report generated: {filepath} ({len(records)} records)")
        return filepath
    
    def get_summary_stats(self, records: List) -> Dict:
        """
        Calculate summary statistics from records for dashboard display.
        
        Returns:
            Dict with keys: total, by_type, avg_confidence, 
            plates_detected, with_evidence
        """
        if not records:
            return {
                'total': 0, 'by_type': {}, 'avg_confidence': 0.0,
                'plates_detected': 0, 'with_evidence': 0
            }
        
        by_type = {}
        total_conf = 0.0
        plates = 0
        evidence = 0
        
        for r in records:
            by_type[r.violation_type] = by_type.get(r.violation_type, 0) + 1
            total_conf += r.confidence
            if r.plate_number:
                plates += 1
            if r.evidence_path:
                evidence += 1
        
        return {
            'total': len(records),
            'by_type': by_type,
            'avg_confidence': total_conf / len(records),
            'plates_detected': plates,
            'with_evidence': evidence
        }
