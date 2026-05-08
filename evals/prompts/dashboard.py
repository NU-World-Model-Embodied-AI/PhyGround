"""Simple dashboard to view prompt YAML files."""

import yaml
from pathlib import Path
from flask import Flask, render_template_string

_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Prompt Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: #fff; padding: 24px 32px; }
  .header h1 { font-size: 1.5rem; font-weight: 600; }
  .header p { color: #aaa; margin-top: 4px; font-size: 0.9rem; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
  .tab { padding: 8px 20px; border: 1px solid #ddd; border-radius: 6px;
         background: #fff; cursor: pointer; font-size: 0.9rem; transition: all 0.15s; }
  .tab:hover { border-color: #667; }
  .tab.active { background: #1a1a2e; color: #fff; border-color: #1a1a2e; }
  .yaml-card { display: none; }
  .yaml-card.active { display: block; }
  .meta { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .meta-item { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
               padding: 12px 20px; }
  .meta-label { font-size: 0.75rem; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }
  .meta-value { font-size: 1rem; font-weight: 600; margin-top: 2px; }
  .section { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
             margin-bottom: 16px; overflow: hidden; }
  .section-header { padding: 12px 20px; background: #fafafa; border-bottom: 1px solid #e0e0e0;
                    font-weight: 600; font-size: 0.95rem; cursor: pointer; user-select: none;
                    display: flex; justify-content: space-between; align-items: center; }
  .section-header:hover { background: #f0f0f0; }
  .section-header .arrow { transition: transform 0.2s; font-size: 0.8rem; color: #888; }
  .section-header.collapsed .arrow { transform: rotate(-90deg); }
  .section-body { padding: 0; }
  .section-header.collapsed + .section-body { display: none; }
  .prompt-row { border-bottom: 1px solid #f0f0f0; }
  .prompt-row:last-child { border-bottom: none; }
  .prompt-key { padding: 10px 20px; background: #f8f9fa; font-weight: 600;
                font-size: 0.85rem; color: #555; border-bottom: 1px solid #eee; }
  .prompt-text { padding: 16px 20px; white-space: pre-wrap; font-family: 'SF Mono', Monaco,
                 'Cascadia Code', monospace; font-size: 0.82rem; line-height: 1.6;
                 color: #2d2d2d; background: #fff; }
  .raw-block { padding: 16px 20px; white-space: pre-wrap; font-family: 'SF Mono', Monaco,
               monospace; font-size: 0.82rem; line-height: 1.5; background: #fafafa; }
</style>
</head>
<body>
<div class="header">
  <h1>Prompt YAML Viewer</h1>
  <p>evals/prompts/ &mdash; {{ files | length }} files</p>
</div>
<div class="container">
  <div class="tabs">
    {% for f in files %}
    <div class="tab {% if loop.first %}active{% endif %}" onclick="switchTab('{{ f.name }}', this)">
      {{ f.name }}
    </div>
    {% endfor %}
  </div>

  {% for f in files %}
  <div id="card-{{ f.name }}" class="yaml-card {% if loop.first %}active{% endif %}">
    <div class="meta">
      <div class="meta-item">
        <div class="meta-label">Scheme</div>
        <div class="meta-value">{{ f.data.get('scheme', '-') }}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">General Keys</div>
        <div class="meta-value">{{ f.data.get('general_keys', []) | join(', ') or '-' }}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Physical Sub-Questions</div>
        <div class="meta-value">{{ f.data.get('physical_sub_questions', '-') }}</div>
      </div>
    </div>

    {% if f.data.get('description') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        Description <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        <div class="raw-block">{{ f.data['description'] }}</div>
      </div>
    </div>
    {% endif %}

    {% if f.data.get('system_prompt') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        System Prompt <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        <div class="prompt-text">{{ f.data['system_prompt'] }}</div>
      </div>
    </div>
    {% endif %}

    {% if f.data.get('eval_prompts') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        Eval Prompts <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        {% for key, val in f.data['eval_prompts'].items() %}
        <div class="prompt-row">
          <div class="prompt-key">{{ key }}</div>
          <div class="prompt-text">{{ val }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if f.data.get('training_prompts') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        Training Prompts <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        {% for key, val in f.data['training_prompts'].items() %}
        <div class="prompt-row">
          <div class="prompt-key">{{ key }}</div>
          <div class="prompt-text">{{ val }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if f.data.get('physical_template') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        Physical Template <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        <div class="prompt-text">{{ f.data['physical_template'] }}</div>
      </div>
    </div>
    {% endif %}

    {% if f.data.get('sub_questions') %}
    <div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        Sub-Questions Config <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body">
        <div class="raw-block">{{ f.sub_questions_str }}</div>
      </div>
    </div>
    {% endif %}

    <div class="section">
      <div class="section-header collapsed" onclick="toggleSection(this)">
        Raw YAML <span class="arrow">&#9660;</span>
      </div>
      <div class="section-body" style="display:none">
        <div class="raw-block">{{ f.raw }}</div>
      </div>
    </div>
  </div>
  {% endfor %}
</div>

<script>
function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.yaml-card').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('card-' + name).classList.add('active');
}
function toggleSection(header) {
  header.classList.toggle('collapsed');
  const body = header.nextElementSibling;
  body.style.display = body.style.display === 'none' ? '' : 'none';
}
</script>
</body>
</html>
"""


def load_yamls():
    files = []
    for p in sorted(_DIR.glob("*.yaml")):
        raw = p.read_text()
        data = yaml.safe_load(raw) or {}
        sq = data.get("sub_questions")
        sq_str = yaml.dump(sq, default_flow_style=False) if sq else ""
        files.append({"name": p.name, "data": data, "raw": raw, "sub_questions_str": sq_str})
    return files


@app.route("/")
def index():
    return render_template_string(TEMPLATE, files=load_yamls())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=True)
