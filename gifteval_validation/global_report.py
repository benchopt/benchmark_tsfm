"""Render precomputed GIFT-Eval global statistics without changing detailed views."""
import html
import pandas as pd


def number(value):
    return "—" if pd.isna(value) else f"{value:.6g}"


def render_global_report(stats, scope):
    escape = lambda value: html.escape(str(value), quote=True)
    metrics = list(scope["panel_ids"])
    models = list(scope["models"])
    metric_options = "".join(f'<option value="{i}">{escape(m)}</option>' for i, m in enumerate(metrics))
    panels = []
    for i, metric in enumerate(metrics):
        selected = stats.loc[stats.metric.eq(metric)]
        bench = selected.loc[selected.system.eq("BenchOpt")].set_index("model_id")
        official = selected.loc[selected.system.eq("GIFT-Eval")].set_index("model_id")
        if (not bench.index.is_unique or not official.index.is_unique
                or set(bench.index) != set(models) or set(official.index) != set(models)):
            raise ValueError("Expected one row per model in each system")
        ordered = bench.reset_index().sort_values(["position", "model_id"], na_position="last")
        rows = []
        for model in ordered.model_id:
            local, ref = bench.loc[model], official.loc[model]
            positions = ["—" if pd.isna(r.position) else str(int(r.position)) for r in [local, ref]]
            cells = [f'<td>{positions[0]}</td>', f'<td>{positions[1]}</td>',
                     f'<th scope="row">{escape(model)}</th>']
            difference = local.mean_value - ref.mean_value
            delta_class = "gift-global-delta"
            if pd.isna(difference):
                delta = "—"
            else:
                percentage = (
                    f"{100 * difference / abs(ref.mean_value):+.2f} %"
                    if abs(ref.mean_value) > 1e-12 else "% non défini"
                )
                delta = f"{difference:+.6g} ({percentage})"
                if difference < 0:
                    delta_class += " gift-global-better"
                elif difference > 0:
                    delta_class += " gift-global-worse"
            cells.extend([
                f'<td>{number(local.mean_value)}</td>',
                f'<td>{number(ref.mean_value)}</td>',
                f'<td class="{delta_class}">{delta}</td>',
                f'<td>{number(local.std_value)}</td>',
                f'<td>{number(ref.std_value)}</td>',
            ])
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        panels.append(f'<div class="gift-global-metric" data-metric="{i}"><h3>{escape(metric)} ↓</h3>'
                      f'<p>Panel commun : {len(scope["panel_ids"][metric])}/{scope["aligned_count"]} '
                      'configurations alignées. Mêmes configurations et mêmes concurrents des deux côtés.</p>'
                      '<div class="gift-global-table" tabindex="0" role="region" '
                      f'aria-label="Comparaison globale {escape(metric)} : tableau défilant">'
                      f'<table><caption>Classement et comparaison — {escape(metric)}</caption><thead><tr>'
                      '<th scope="col">Position BenchOpt</th>'
                      '<th scope="col">Position GIFT-Eval</th>'
                      '<th scope="col">Modèle</th><th scope="col">Moyenne BenchOpt</th>'
                      '<th scope="col">Moyenne GIFT-Eval</th><th scope="col">Écart</th>'
                      '<th scope="col">Écart-type BenchOpt</th><th scope="col">Écart-type GIFT-Eval</th>' +
                      '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></div>')
    return f"""
<section id="global-panel" class="panel"><div class="card">
<h2>Synthèse globale — Global Performance Summary</h2>
<p>{scope['aligned_count']}/{scope['reference_count']} configurations officielles alignées.
Quatre architectures : Chronos, Chronos2, Toto-2.0 et Naive.
Les résultats officiels affichés ici sont recalculés sur notre panel commun,
pas sur l’ensemble du leaderboard officiel.</p>
<div class="controls"><label>Métrique <select id="gift-global-metric">{metric_options}</select></label>
</div>
<p>Positions calculées selon la valeur moyenne de la métrique, par ordre croissant
(plus faible = meilleur), indépendamment dans chaque système.
Les moyennes exactement égales partagent la même position (1, 1, 3).
Les lignes sont triées par position BenchOpt.</p>
<p><strong>Écart = moyenne BenchOpt − moyenne GIFT-Eval</strong>, calculé avant arrondi.
Entre parenthèses : 100 × écart / |moyenne GIFT-Eval|.
Vert : erreur moyenne plus faible ; rouge : erreur moyenne plus élevée ; zéro : neutre.
Le pourcentage est indéfini lorsque la référence est nulle ou presque nulle.
Un écart négatif indique une erreur moyenne plus faible côté BenchOpt.
Les deux écarts-types décrivent séparément la dispersion des valeurs entre configurations.</p>
{''.join(panels)}
<details><summary>Périmètre et conventions scientifiques</summary>
<p>Une configuration ne contribue que si les quatre modèles ont des valeurs finies
dans les deux systèmes. Une absence retire cette configuration de tout le panel
pour la métrique concernée. Le nombre de configurations retenues est indiqué
au-dessus du tableau. Les exclusions figurent dans
<a href="../artifacts/global_statistics_exclusions.csv">le CSV des exclusions</a>.</p>
<p>Moyenne arithmétique des valeurs brutes ; écart-type population (ddof=0),
pas l’écart-type des rangs. Les échelles des datasets peuvent dominer la moyenne
et la dispersion. Chaque configuration compte une fois : plusieurs horizons d’un
même dataset comptent séparément. Aucune moyenne entre métriques.
Aucune valeur manquante n’est remplacée par zéro. Avec zéro observation,
valeurs et positions indéfinies (—) ; avec une observation, écart-type zéro.</p>
<p><a href="../artifacts/global_statistics.csv">Télécharger la synthèse CSV</a></p>
</details></div></section>
<style>
#global-panel table {{ min-width: 950px; border-collapse: separate; border-spacing: 0; }}
#global-panel caption {{ text-align: left; padding: 10px 0; font-weight: bold; }}
#global-panel th, #global-panel td {{ white-space: nowrap; }}
#global-panel thead th[colspan] {{ text-align: center; }}
#global-panel .gift-global-delta {{ background: #edf2fc; font-weight: 600; border-right: 2px solid #ccd6e7; }}
.gift-global-table {{ overflow-x: auto; margin: 16px 0; }}
#global-panel .gift-global-better {{ color: #166534; background: #ecfdf3; }}
#global-panel .gift-global-worse {{ color: #991b1b; background: #fff1f2; }}
#global-panel p {{ line-height: 1.6; }}
#global-panel select {{ margin: 8px; padding: 10px; max-width: 100%; }}

</style>
<script>
(() => {{
 const root = document.getElementById('global-panel');
 const metric = document.getElementById('gift-global-metric');
 const update = () => {{
  root.querySelectorAll('.gift-global-metric').forEach(p => {{ p.hidden = p.dataset.metric !== metric.value; }});
 }};
 metric.addEventListener('change', update); update();
}})();
</script>
"""
