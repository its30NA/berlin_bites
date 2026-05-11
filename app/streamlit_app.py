import streamlit as st
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title='Berlin Bites', layout='wide')
st.title('🍽️ Berlin Bites — Dev UI')
st.write('A small developer interface to run project scripts and open notebooks.')

notebooks_dir = REPO_ROOT / 'notebooks'
if notebooks_dir.exists():
    st.subheader('Notebooks')
    for nb in sorted(notebooks_dir.glob('*.ipynb')):
        st.write(f"- {nb.name}")

st.subheader('Scripts')

def run_cmd(cmd, timeout=600):
    try:
        res = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, '', str(e)

col1, col2 = st.columns(2)
with col1:
    if st.button('Run data_collection.py'):
        with st.spinner('Running data_collection.py...'):
            code, out, err = run_cmd('python src/data_collection.py')
            st.code(out or '(no stdout)')
            if err:
                st.error(err)

    if st.button('Run preprocessing01.py'):
        with st.spinner('Running preprocessing01.py...'):
            code, out, err = run_cmd('python src/preprocessing01.py')
            st.code(out or '(no stdout)')
            if err:
                st.error(err)

with col2:
    if st.button('Run preprocessing.py'):
        with st.spinner('Running preprocessing.py...'):
            code, out, err = run_cmd('python src/preprocessing.py')
            st.code(out or '(no stdout)')
            if err:
                st.error(err)

    if st.button('Run train_model.py (may take long)'):
        with st.spinner('Running train_model.py...'):
            code, out, err = run_cmd('python src/train_model.py')
            st.code(out or '(no stdout)')
            if err:
                st.error(err)

st.markdown('---')
st.write('Files in project root:')
for p in sorted(REPO_ROOT.glob('*')):
    st.write('-', p.name)

