import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os


st.set_page_config(
    page_title="Human-centered AI Language Technology",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Human-centered AI Language Technology")

# Define sample text
sample_text = """nobis aegrus modi minima aut voluptatem. Necessitatibus aut nam quis porro et a. Interdum et malesuada fames ac ante ipsum primis in faucibus. Nullam pulvinar pellentesque. Nullam quis luctus diam, a feugiat enim. Aenean et tempor ante, non hendrerit tellus. Fusce finibus mauris ipsum, tincidunt semper turpis consectetur eget."""

# Sidebar configuration
with st.sidebar:
    st.title("Configuración de Traducción")
    
    
    # LLM model selector
    llm_model = st.selectbox(
        "Selecciona el Modelo LLM",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
    )
    
    # API Key input
    api_key = st.text_input("Ingresa tu API Key de OpenAI", type="password")
    
    # Temperature sliders
    st.subheader("Configuración de Temperatura")
    temp_basic = st.slider("Temperatura del traductor basico", 0.0, 2.0, 0.7, 0.01)
    temp_agent_translator = st.slider("Temperatura del agente traductor", 0.0, 2.0, 0.7, 0.01)
    temp_agent_ple = st.slider("Temperatura de Plain Language Editor", 0.0, 2.0, 1.0, 0.01)
    temp_agent_quality = st.slider("Temperatura de Quality Estimator", 0.0, 2.0, 0.7, 0.01)

# Main area


# File uploader
uploaded_file = st.file_uploader("Subir archivo de texto (.txt)", type=["txt"], help="Limite 20MB per file + TXT")

# Text input area
if uploaded_file:
    input_text = uploaded_file.getvalue().decode("utf-8")
    st.text_area(
        "Texto de Entrada:",
        value=input_text,
        height=200,        
        placeholder="aqui saldra el texto que se ha subido"
    )

# Translation button
translate_button = st.button("boton para ejecutar la traduccion")

# Results area
if translate_button and input_text and api_key:
    # Initialize LLM models with different temperatures
    llm_basic_tranlator = ChatOpenAI(
        model=llm_model,
        temperature=temp_basic,
        api_key=api_key
    )
    
    llm_agent_translator = ChatOpenAI(
        model=llm_model,
        temperature=temp_agent_translator,
        api_key=api_key
    )
    
    llm_agent_plain_editor = ChatOpenAI(
        model=llm_model,
        temperature=temp_agent_ple,
        api_key=api_key
    )
    
    llm_agent_quality_estimator = ChatOpenAI(
        model=llm_model,
        temperature=temp_agent_quality,
        api_key=api_key
    )
    
    # Create LCEL chains
    # Basic LLM Translator (Agent B)
    basic_translator_prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            (
                                "Eres un agente de traducción automática. Tu tarea es traducir el siguiente texto al inglés. Contesta siempre solo con la traducción, sin nigun comentario u observacion adicional."
                            )
                        ),
                        (
                            "human",
                            "Traduce el siguiente texto al inglés: {text}. Debes mantener el significado original del texto. Devuelve como resultado solo la traducción, sin nigun comentario u observacion adicional."
                            
                        )
                    ])
    
    basic_translator_chain = basic_translator_prompt_template | llm_basic_tranlator | StrOutputParser()
    
    agent_translator_prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            (
                                "Actúa como un traductor médico especializado con fluidez nativa en español e inglés. Tu tarea es convertir textos médicos técnicos del español de España (castellano) al inglés estadounidense, manteniendo fidelidad semántica, tecnicismos y estructura original.\n\nInstrucciones Clave:\n\nPrecisión Terminológica: Usa equivalentes validados (Ej: \'hipertensión arterial\' → \'hypertension\', \'taquicardia sinusal\' → \'sinus tachycardia\').\n\nConservar Formatos: Mantén siglas (Ej: HTA → HTN), códigos CIE-10, valores numéricos (Ej: 160 mg/dL) y estructura del documento (secciones, viñetas).\n\nContexto Clínico: Prioriza equivalentes aceptados en literatura médica anglófona (Ej: \'edema maleolar\' → \'ankle edema\', no \'swelling\').\n\nRegistros y Normas:\n\nMedicamentos: Conserva nombres científicos (Ej: \'enalapril\' → \'enalapril\', no marcas comerciales).\n\nFechas: Convierte formatos (Ej: \'15 de Octubre de 2023\' → \'October 15, 2023\').\n\nAmbigüidades: Si un término tiene múltiples traducciones (Ej: \'disnea\' → \'dyspnea\' o \'shortness of breath\'), elige la opción más frecuente en contextos formales.\n\nNotas Adicionales:\n\nEvita interpretaciones o resúmenes.\n\nSeñala entre corchetes [ ] cualquier incertidumbre en la traducción. Contesta siempre solo con la traducción, sin nigun comentario u observacion adicional."
                            )
                        ),
                        (
                            "human",
                            "Traduce el siguiente informe médico al inglés, conservando todos los detalles técnicos y estructura. Aquí el texto: {text}."
                            
                        )
                    ])
    
    agent_translator_chain= agent_translator_prompt_template | llm_agent_translator | StrOutputParser()
    
    # # Plain Language Editor
    agent_plain_editor_prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            (
                                "Actúa como un traductor médico especializado con habilidades de adaptación para públicos no expertos. Tu tarea es procesar textos médicos. Recibiras dos textos. El primero es el texto original en castellano. El segundo es la traducción inicial al inglés. Tu tarea es simplificar la traducción al inglés para que sea comprensible para un público general, manteniendo la fidelidad semántica y la información esencial.\n\nInstrucciones Clave:\n\nClaridad: Usa un lenguaje sencillo y directo, evitando tecnicismos y jerga médica.\n\nEstructura: Organiza la información en párrafos cortos y secuencias lógicas.\n\nPrecisión: Conserva la información esencial y evita interpretaciones o resúmenes.\n\nAdaptación: Ajusta el tono y estilo para un público no experto, sin perder la rigurosidad del contenido.\n\nNotas Adicionales:\n\nEvita añadir información nueva o modificar el contenido original.\n\nDevuelve solo la versión simplificada, sin nigun comentario u observacion adicional. Explica conceptos complejos sin alterar hechos (Ej: 'taquicardia sinusal' → 'fast heartbeat originating from the heart’s natural pacemaker').\n\nSustituye latinismos por términos comunes (Ej: 'disnea' → 'shortness of breath').\n\nMantén números y códigos, pero añade contexto (Ej: 'LDL: 160 mg/dL' → 'bad cholesterol level (160 mg/dL) – above normal').\n\nUsa oraciones cortas y voz activa.\n\nProhibido:\n\nInterpretar diagnósticos o añadir información no explícita.\n\nUsar metáforas o lenguaje subjetivo.\n\nAñadir información no presente en el texto original.\n\nContesta siempre solo con la versión simplificada, sin nigun comentario u observacion adicional."
                            )
                        ),
                        (
                            "human",
                            "Procesa este informe médico siguiendo las instrucciones. Aquí el texto original en castellano: {original_text}. Y aquí la traducción inicial al inglés: {translated_text}."
                            
                        )
                    ])
    
    agent_plain_editor_chain =  agent_plain_editor_prompt_template | llm_agent_plain_editor | StrOutputParser()
    
    # # Quality Estimator
    agent_quality_estimator_prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            (
                                """Actúa como un evaluador experto en traducción médica bilingüe (español-inglés). Tu tarea es analizar una versión simplificada en inglés comparándola con:
                                Texto original en español
                                Traducción técnica inicial
                                
                                Criterios de Evaluación:
                                Exactitud (40%): ¿La simplificación mantiene TODOS los hechos médicos del original sin distorsiones?
                                
                                Claridad (30%): ¿El lenguaje es accesible pero técnicamente correcto?
                                
                                Conservación de Datos (20%): ¿Mantiene códigos (CIE-10), valores numéricos y jerarquías?
                                
                                Errores Graves (10%): ¿Hay omisiones/adiciones no justificadas o términos mal adaptados?
                                
                                Instrucciones:
                                Penaliza con -15% por cada error factual o dato crítico alterado.
                                
                                Si el score final es <50%, considera la traducción como no confiable.
                                
                                Devuelve SOLO el porcentaje numérico (Ej: '82%') sin comentarios."""
                            )
                        ),
                        (
                            "human",
                            """
                            Evalúa esta traducción simplificada: {simplified_text}   
                            Texto original en español: {original_text}
                            Traducción técnica inicial: {translated_text}
                            Devuelve solo el score de credibilidad entre 0-100%
                            """
                            
                        )
                    ])
    
    agent_quality_estimator_chain = agent_quality_estimator_prompt_template | llm_agent_quality_estimator | StrOutputParser()
    
    # Execute translations with progress indicators
    with st.spinner("Ejecutando traducción básica..."):
        basic_translation = basic_translator_chain.invoke(input_text)
    
    with st.spinner("Ejecutando el agente traductor..."):
        agent_translation = agent_translator_chain.invoke(input_text)

    with st.spinner("Ejecutando Plain Language Editor..."):
        simplified_translation = agent_plain_editor_chain.invoke({
            "original_text": input_text,
            "translated_text": agent_translation
        })

    with st.spinner("Ejecutando Quality Estimator..."):
        quality_score = agent_quality_estimator_chain.invoke({
            "original_text": input_text,
            "translated_text": agent_translation,
            "simplified_text": simplified_translation
        })

        print(quality_score)
        

        # Ensure the quality score is a number
    
    # with st.spinner("Evaluando calidad de traducción..."):
    #     quality_score = quality_chain.invoke({
    #         "original_text": input_text,
    #         "translated_text": classic_translation,
    #         "simplified_text": simplified_translation
    #     })
        # Ensure the quality score is a number
        # try:
        #     quality_score = float(quality_score.strip())
        #     quality_score_formatted = f"{quality_score:.4f}"
        # except ValueError:
        #     quality_score_formatted = "Error: No se pudo calcular la puntuación"
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Traduccion fase 1")
        st.text_area(
            "Resultado:",
            value=basic_translation,
            height=600,
            placeholder="aqui saldra el resultado de la traduccion del agente B"
        )
    
    with col2:
        st.subheader("Traduccion final")
        st.text_area(
            "Agente traductor:",
            value=agent_translation,
            height=600)
        
    st.subheader("Plain Language Editor")
    st.text_area(
        "Resultado:",
        value=simplified_translation,
        height=600,
        placeholder="aqui saldra el resultado de la traduccion tras ser procesada por el Plan Language editor"
    )
                  
    st.subheader("Quality Estimator")
        
    st.metric("Puntuación de Calidad", quality_score)

elif translate_button and not api_key:
    st.error("Por favor, ingresa tu API Key de OpenAI para continuar.")
elif translate_button and not input_text:
    st.warning("Por favor, sube un archivo o ingresa texto para traducir.")