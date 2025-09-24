from flask import Flask, request, jsonify
import numpy as np
import re
import math
import joblib
import os

# CORREÇÃO: Importar sklearn ANTES de usar RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

class AIDetectorComRelatorio:
    def __init__(self):
        # AGORA o RandomForestClassifier está disponível
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        self.carregar_ou_treinar_modelo()
        
        # Padrões linguísticos típicos de IA
        self.padroes_ia = {
            'expressoes_formais': [
                r'é importante destacar', r'convém ressaltar', r'considerando os fatores',
                r'pode-se inferir', r'é fundamental observar', r'observa-se que',
                r'conclui-se que', r'de maneira clara e objetiva', r'abordagem sistemática',
                r'o presente estudo', r'de acordo com', r'vale ressaltar', r'verifica-se que',
                r'cabe salientar', r'é pertinente mencionar', r'pressupõe-se que'
            ],
            'conectivos_complexos': [
                r'portanto', r'consequentemente', r'adicionalmente', r'notavelmente',
                r'consideravelmente', r'significativamente', r'efetivamente'
            ],
            'estruturas_passivas': [
                r'é realizado', r'é observado', r'é verificado', r'é constatado',
                r'é possível identificar', r'pode ser observado', r'deve ser considerado'
            ],
            'palavras_superlativas': [
                r'extremamente', r'altamente', r'profundamente', r'intensamente',
                r'significativamente', r'consideravelmente', r'notavelmente'
            ]
        }
    
    def tokenizacao_simples(self, texto):
        palavras = re.findall(r'\b\w+\b', texto.lower())
        return palavras
    
    def dividir_frases(self, texto):
        frases = re.split(r'[.!?]+(?:\s+|$)', texto)
        return [f.strip() for f in frases if f.strip()]
    
    def analisar_termos_suspeitos(self, texto):
        """Analisa o texto e identifica termos/expressões suspeitos de IA"""
        termos_detectados = []
        texto_lower = texto.lower()
        
        # Analisar expressões formais típicas de IA
        for expressao in self.padroes_ia['expressoes_formais']:
            matches = list(re.finditer(expressao, texto_lower))
            for match in matches:
                start, end = match.span()
                contexto = texto[max(0, start-50):min(len(texto), end+50)]
                termos_detectados.append({
                    'termo': texto[match.start():match.end()],
                    'tipo': 'expressao_formal',
                    'justificativa': 'Expressão excessivamente formal comum em textos de IA',
                    'contexto': contexto,
                    'posicao': (start, end)
                })
        
        # Analisar conectivos complexos
        for conectivo in self.padroes_ia['conectivos_complexos']:
            if re.search(r'\b' + conectivo + r'\b', texto_lower):
                matches = list(re.finditer(r'\b' + conectivo + r'\b', texto_lower))
                for match in matches:
                    termos_detectados.append({
                        'termo': texto[match.start():match.end()],
                        'tipo': 'conectivo_complexo',
                        'justificativa': 'Uso frequente de conectivos complexos típico de IA',
                        'contexto': texto[max(0, match.start()-30):min(len(texto), match.end()+30)],
                        'posicao': (match.start(), match.end())
                    })
        
        # Analisar estruturas passivas
        for estrutura in self.padroes_ia['estruturas_passivas']:
            matches = list(re.finditer(estrutura, texto_lower))
            for match in matches:
                termos_detectados.append({
                    'termo': texto[match.start():match.end()],
                    'tipo': 'voz_passiva',
                    'justificativa': 'Uso excessivo de voz passiva, comum em textos formais de IA',
                    'contexto': texto[max(0, match.start()-40):min(len(texto), match.end()+40)],
                    'posicao': (match.start(), match.end())
                })
        
        # Analisar palavras superlativas
        for superlativo in self.padroes_ia['palavras_superlativas']:
            matches = list(re.finditer(r'\b' + superlativo + r'\b', texto_lower))
            for match in matches:
                termos_detectados.append({
                    'termo': texto[match.start():match.end()],
                    'tipo': 'superlativo',
                    'justificativa': 'Uso frequente de intensificadores e superlativos',
                    'contexto': texto[max(0, match.start()-25):min(len(texto), match.end()+25)],
                    'posicao': (match.start(), match.end())
                })
        
        return termos_detectados
    
    def gerar_texto_destacado(self, texto, termos_detectados):
        """Gera versão do texto com termos suspeitos destacados"""
        if not termos_detectados:
            return texto
            
        texto_destacado = texto
        # Ordenar termos por posição (do final para o início para evitar problemas com índices)
        termos_ordenados = sorted(termos_detectados, key=lambda x: x['posicao'][0], reverse=True)
        
        for termo_info in termos_ordenados:
            start, end = termo_info['posicao']
            termo_original = texto[start:end]
            
            # Criar span com destaque
            span_destacado = f'<span class="termo-suspeito" data-tipo="{termo_info["tipo"]}" title="{termo_info["justificativa"]}">{termo_original}</span>'
            
            # Inserir no texto
            texto_destacado = texto_destacado[:start] + span_destacado + texto_destacado[end:]
        
        return texto_destacado
    
    def extrair_features(self, texto):
        texto_analise = texto
        if len(texto) > 5000:
            texto_analise = texto[:2000] + texto[len(texto)//2-500:len(texto)//2+500] + texto[-2000:]
        
        palavras = self.tokenizacao_simples(texto_analise)
        frases = self.dividir_frases(texto_analise)
        
        features = {}
        features['comprimento'] = len(texto)
        features['num_palavras'] = len(palavras)
        features['num_frases'] = len(frases)
        
        if palavras:
            features['comp_medio_palavra'] = float(np.mean([len(p) for p in palavras]))
            features['palavras_por_frase'] = float(len(palavras) / len(frases)) if frases else 0.0
            features['diversidade_lexical'] = float(len(set(palavras)) / len(palavras))
        else:
            features['comp_medio_palavra'] = 0.0
            features['palavras_por_frase'] = 0.0
            features['diversidade_lexical'] = 0.0
        
        features['palavras_longas'] = float(len([p for p in palavras if len(p) > 6]) / len(palavras)) if palavras else 0.0
        features['formalidade'] = float(self.calcular_formalidade(texto_analise))
        
        return features
    
    def calcular_formalidade(self, texto):
        palavras_formais = ['portanto', 'consequentemente', 'adicionalmente', 'fundamental']
        palavras_informais = ['tipo', 'assim', 'ok', 'bem', 'acho', 'tá']
        
        texto_lower = texto.lower()
        formal = sum(1 for p in palavras_formais if p in texto_lower)
        informal = sum(1 for p in palavras_informais if p in texto_lower)
        
        total = len(self.tokenizacao_simples(texto))
        return (formal - informal) / total if total > 0 else 0.0
    
    def carregar_ou_treinar_modelo(self):
        try:
            if os.path.exists('modelo_web.pkl'):
                self.model = joblib.load('modelo_web.pkl')
                self.is_trained = True
                print("✅ Modelo carregado")
                return
        except:
            print("ℹ️  Modelo não encontrado, treinando novo...")
        
        self.treinar_modelo()
    
    def treinar_modelo(self):
        print("🔧 Treinando modelo...")
        
        humanos = [
            "Fui na padaria e comprei pão. O padeiro foi muito simpático!",
            "Não acredito que esqueci minha carteira em casa. Que chato!",
            "Meu time ganhou o jogo de virada. Foi emocionante demais!",
            "Estou com uma fome danada. Vou pedir uma pizza bem grande.",
            "O trânsito hoje estava impossível. Levei duas horas pra chegar."
        ]
        
        ia_textos = [
            "É importante destacar que a eficácia do processo depende de diversos fatores inter-relacionados.",
            "Considerando os aspectos mencionados anteriormente, pode-se concluir que a abordagem é adequada.",
            "Observa-se que a implementação das estratégias resulta em benefícios significativos.",
            "Convém ressaltar que a metodologia utilizada segue os padrões estabelecidos.",
            "Conclui-se que a proposta apresenta viabilidade técnica e operacional."
        ]
        
        textos = humanos + ia_textos
        labels = [0] * len(humanos) + [1] * len(ia_textos)
        
        features = []
        for texto in textos:
            feat = self.extrair_features(texto)
            features.append(list(feat.values()))
        
        try:
            self.model.fit(features, labels)
            self.is_trained = True
            joblib.dump(self.model, 'modelo_web.pkl')
            print("✅ Modelo treinado com sucesso")
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            self.is_trained = False
    
    def predict(self, texto):
        if not texto or len(texto.strip()) < 20:
            return {
                'ai_probability': 0.5, 
                'human_probability': 0.5, 
                'confidence': 0.1,
                'termos_suspeitos': [],
                'texto_destacado': texto
            }
        
        try:
            # Análise de termos suspeitos
            termos_suspeitos = self.analisar_termos_suspeitos(texto)
            texto_destacado = self.gerar_texto_destacado(texto, termos_suspeitos)
            
            # Análise tradicional
            feat_dict = self.extrair_features(texto)
            features = [float(v) for v in feat_dict.values()]
            
            if self.is_trained:
                proba = self.model.predict_proba([features])[0]
                prob_ia = float(proba[1])
                confianca = float(np.max(proba))
            else:
                # Fallback heurístico
                prob_ia = 0.3 if feat_dict['formalidade'] > 0.1 else 0.7
                confianca = 0.6
            
            # Agrupar termos por tipo para relatório
            termos_por_tipo = {}
            for termo in termos_suspeitos:
                if termo['tipo'] not in termos_por_tipo:
                    termos_por_tipo[termo['tipo']] = []
                termos_por_tipo[termo['tipo']].append(termo)
            
            resultado = {
                'ai_probability': round(prob_ia, 3),
                'human_probability': round(1 - prob_ia, 3),
                'confidence': round(confianca, 2),
                'text_analyzed_length': len(texto),
                'termos_suspeitos': termos_suspeitos,
                'texto_destacado': texto_destacado,
                'estatisticas_deteccao': {
                    'total_termos': len(termos_suspeitos),
                    'termos_por_tipo': {tipo: len(termos) for tipo, termos in termos_por_tipo.items()},
                    'densidade_termos': len(termos_suspeitos) / len(texto.split()) if texto.split() else 0
                }
            }
            
            return resultado
            
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return {
                'ai_probability': 0.5, 
                'human_probability': 0.5, 
                'confidence': 0.1,
                'termos_suspeitos': [],
                'texto_destacado': texto
            }

# CORREÇÃO: Inicializar DEPOIS da definição da classe
detector = AIDetectorComRelatorio()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Detector de IA - Análise Detalhada</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; padding: 20px; 
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; color: white; margin-bottom: 40px; }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .app-card { 
                background: white; border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1); padding: 40px; 
                margin-bottom: 20px;
            }
            .text-area { 
                width: 100%; min-height: 200px; padding: 20px; 
                border: 2px solid #e2e8f0; border-radius: 10px; 
                font-size: 1rem; resize: vertical; margin-bottom: 20px;
                font-family: monospace; line-height: 1.6;
            }
            .analyze-btn { 
                background: #2563eb; color: white; border: none; 
                padding: 15px 30px; border-radius: 10px; font-size: 1.1rem; 
                cursor: pointer; width: 100%; margin-bottom: 20px;
            }
            .analyze-btn:disabled { background: #94a3b8; cursor: not-allowed; }
            .result { margin-top: 20px; padding: 20px; background: #f8fafc; border-radius: 10px; }
            .stats { background: #e2e8f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
            
            /* Estilos para termos suspeitos */
            .termo-suspeito {
                background-color: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 3px;
                padding: 2px 4px;
                margin: 0 1px;
                cursor: help;
                position: relative;
            }
            .termo-suspeito:hover {
                background-color: #fde68a;
                z-index: 1000;
            }
            .termo-suspeito[data-tipo="expressao_formal"] { border-color: #ef4444; background-color: #fef2f2; }
            .termo-suspeito[data-tipo="conectivo_complexo"] { border-color: #8b5cf6; background-color: #faf5ff; }
            .termo-suspeito[data-tipo="voz_passiva"] { border-color: #06b6d4; background-color: #ecfeff; }
            .termo-suspeito[data-tipo="superlativo"] { border-color: #f59e0b; background-color: #fffbeb; }
            
            .texto-destacado {
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                line-height: 1.8;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .relatorio-termos {
                margin-top: 20px;
            }
            
            .tipo-termo {
                background: #f8fafc;
                border-left: 4px solid #2563eb;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
            
            .termo-item {
                background: white;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                border-left: 3px solid #94a3b8;
            }
            
            .legenda {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 15px 0;
            }
            
            .legenda-item {
                display: flex;
                align-items: center;
                gap: 5px;
                padding: 5px 10px;
                background: #f8fafc;
                border-radius: 5px;
                font-size: 0.9em;
            }
            
            .legenda-cor {
                width: 15px;
                height: 15px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Detector de IA - Análise Detalhada</h1>
                <p>Identifique termos e expressões suspeitos de terem sido gerados por IA</p>
            </div>
            
            <div class="app-card">
                <h3>📝 Cole seu texto para análise</h3>
                <textarea class="text-area" id="textInput" placeholder="Cole o texto que deseja analisar aqui..."></textarea>
                
                <div class="stats" id="textStats">
                    Caracteres: 0 | Palavras: 0 | Frases: 0
                </div>
                
                <button class="analyze-btn" onclick="analyzeText()" id="analyzeBtn">Analisar Texto</button>
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('textInput').addEventListener('input', function() {
                const text = this.value;
                const chars = text.length;
                const words = text.trim() ? text.trim().split(/\\s+/).length : 0;
                const sentences = text.split(/[.!?]+/).length - 1;
                document.getElementById('textStats').innerHTML = 
                    `Caracteres: ${chars} | Palavras: ${words} | Frases: ${sentences}`;
            });
        
            async function analyzeText() {
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                const button = document.getElementById('analyzeBtn');
                
                if (text.length < 20) {
                    resultDiv.innerHTML = '<p style="color: red;">⚠️ Texto muito curto (mínimo 20 caracteres)</p>';
                    return;
                }
                
                button.disabled = true;
                button.textContent = 'Analisando...';
                resultDiv.innerHTML = '<p>⏳ Analisando texto e identificando padrões suspeitos...</p>';
                
                try {
                    const response = await fetch('/api/detect', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        const result = data.result;
                        const aiProb = Math.round(result.ai_probability * 100);
                        const humanProb = Math.round(result.human_probability * 100);
                        const confidence = Math.round(result.confidence * 100);
                        
                        let color = '#64748b';
                        let veredito = 'Indeterminado';
                        if (aiProb > 70) {
                            color = '#ef4444';
                            veredito = 'Provavelmente IA';
                        } else if (humanProb > 70) {
                            color = '#10b981';
                            veredito = 'Provavelmente Humano';
                        }
                        
                        let relatorioHTML = gerarRelatorioDetalhado(result);
                        
                        resultDiv.innerHTML = `
                            <div style="text-align: center; margin-bottom: 20px;">
                                <h3 style="color: ${color}">${veredito}</h3>
                                <p>🤖 Probabilidade de IA: <strong style="font-size: 1.2em;">${aiProb}%</strong></p>
                                <p>👤 Probabilidade Humana: <strong style="font-size: 1.2em;">${humanProb}%</strong></p>
                                <p>📊 Confiança da análise: ${confidence}%</p>
                            </div>
                            ${relatorioHTML}
                        `;
                    } else {
                        resultDiv.innerHTML = `<p style="color: red;">❌ Erro: ${data.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<p style="color: red;">❌ Erro de conexão. Tente novamente.</p>';
                } finally {
                    button.disabled = false;
                    button.textContent = 'Analisar Texto';
                }
            }
            
            function gerarRelatorioDetalhado(result) {
                const termos = result.termos_suspeitos || [];
                
                if (termos.length === 0) {
                    return `
                        <div class="relatorio-termos">
                            <h4>✅ Nenhum padrão suspeito detectado</h4>
                            <p>O texto não apresenta os padrões típicos de IA analisados.</p>
                        </div>
                    `;
                }
                
                // Agrupar termos por tipo
                const termosPorTipo = {};
                termos.forEach(termo => {
                    if (!termosPorTipo[termo.tipo]) {
                        termosPorTipo[termo.tipo] = [];
                    }
                    termosPorTipo[termo.tipo].push(termo);
                });
                
                // Gerar HTML para cada tipo
                let tiposHTML = '';
                for (const [tipo, termosTipo] of Object.entries(termosPorTipo)) {
                    let termosHTML = termosTipo.map(termo => `
                        <div class="termo-item">
                            <strong>"${termo.termo}"</strong> - ${termo.justificativa}
                            <br><small>Contexto: "...${termo.contexto}..."</small>
                        </div>
                    `).join('');
                    
                    const tipoLabel = {
                        'expressao_formal': 'Expressões Formais',
                        'conectivo_complexo': 'Conectivos Complexos',
                        'voz_passiva': 'Voz Passiva',
                        'superlativo': 'Superlativos'
                    }[tipo] || tipo;
                    
                    tiposHTML += `
                        <div class="tipo-termo">
                            <h5>${tipoLabel} (${termosTipo.length} ocorrências)</h5>
                            ${termosHTML}
                        </div>
                    `;
                }
                
                return `
                    <div class="relatorio-termos">
                        <h4>🔍 Relatório de Análise Detalhada</h4>
                        <p><strong>${termos.length} termos/padrões suspeitos detectados</strong></p>
                        
                        <div class="legenda">
                            <div class="legenda-item">
                                <div class="legenda-cor" style="background-color: #fef2f2; border: 1px solid #ef4444;"></div>
                                <span>Expressões Formais</span>
                            </div>
                            <div class="legenda-item">
                                <div class="legenda-cor" style="background-color: #faf5ff; border: 1px solid #8b5cf6;"></div>
                                <span>Conectivos Complexos</span>
                            </div>
                            <div class="legenda-item">
                                <div class="legenda-cor" style="background-color: #ecfeff; border: 1px solid #06b6d4;"></div>
                                <span>Voz Passiva</span>
                            </div>
                            <div class="legenda-item">
                                <div class="legenda-cor" style="background-color: #fffbeb; border: 1px solid #f59e0b;"></div>
                                <span>Superlativos</span>
                            </div>
                        </div>
                        
                        <div class="texto-destacado">
                            <h5>Texto com termos destacados:</h5>
                            <div>${result.texto_destacado}</div>
                        </div>
                        
                        <div style="margin-top: 20px;">
                            <h5>📋 Detalhamento dos padrões detectados:</h5>
                            ${tiposHTML}
                        </div>
                    </div>
                `;
            }
            
            // Exemplo de texto para teste
            document.getElementById('textInput').value = `É importante destacar que a implementação de estratégias eficazes requer uma abordagem sistemática e coordenada. Considerando os fatores mencionados anteriormente, pode-se inferir que a solução proposta apresenta viabilidade técnica e operacional. Observa-se que os resultados obtidos são extremamente significativos e demonstram claramente a eficácia da metodologia empregada.

Por outro lado, quando fui ao mercado hoje, o padeiro foi muito simpático comigo. Ele me deu um café e a gente conversou um pouco sobre o time de futebol. Achei bem legal isso, sabe?`;

            document.getElementById('textInput').dispatchEvent(new Event('input'));
        </script>
    </body>
    </html>
    '''

@app.route('/api/detect', methods=['POST'])
def detect_ai():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if len(text) < 20:
            return jsonify({
                'error': 'Texto muito curto. Mínimo 20 caracteres.',
                'min_length': 20
            }), 400
        
        result = detector.predict(text)
        
        return jsonify({
            'success': True,
            'result': result,
            'text_stats': {
                'original_length': len(text),
                'analyzed_length': result.get('text_analyzed_length', len(text)),
                'word_count': len(text.split()),
                'sentences': len(detector.dividir_frases(text))
            }
        })
        
    except Exception as e:
        print(f"💥 ERRO NA API: {e}")
        return jsonify({
            'error': f'Erro na análise: {str(e)}'
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'online', 
        'model': 'active',
        'features': 'relatorio_detalhado',
        'version': '2.0_corrigido'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Detector de IA com Relatório Detalhado - Porta {port}")
    app.run(host='0.0.0.0', port=port)
