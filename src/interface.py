import asyncio
import base64
import logging
import sys
import time
import warnings
from collections.abc import Generator
from pathlib import Path

import nest_asyncio
import streamlit as st

# from src.agents.assistant_agent import stream_messages, update_memory
from src.agents.orchestrator_agent import stream_orchestrated as stream_messages, update_memory
from src.core.database import database_connect
from src.preprocessing.document_parser import parse_document
from src.preprocessing.document_processor import process_and_embed_document

# Configure logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Allow nested event loops (needed for Streamlit + async)
nest_asyncio.apply()

# Suppress unclosed event loop warnings (expected with nest_asyncio + streamlit)
warnings.filterwarnings('ignore', message='unclosed event loop')

DATA_DIR = Path(__file__).parent.parent / 'data'


# Create a single persistent event loop for the session
@st.cache_resource
def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = get_event_loop()
    return loop


async def get_indexed_documents() -> list[str]:
    """Fetch distinct document names from the database."""
    async with database_connect() as pool:
        rows = await pool.fetch(
            'SELECT DISTINCT folder FROM repo ORDER BY folder'
        )
        return [row['folder'] for row in rows]


async def delete_document(doc_name: str) -> None:
    """Delete all embeddings for a document from the database."""
    async with database_connect() as pool:
        await pool.execute('DELETE FROM repo WHERE folder = $1', doc_name)


async def stream_response(
    prompt: str,
    selected_docs: list[str] | None = None,
    message_history: list[dict[str, str]] | None = None,
    memory: str | None = None,
) -> tuple[list[str], list[str]]:
    """Stream response from agent, returning text chunks and any generated images."""
    message_chunks: list[str] = []
    images: list[str] = []

    async for chunk in stream_messages(prompt, selected_docs, message_history, memory):
        if isinstance(chunk, dict) and 'images' in chunk:
            # This is the images payload at the end
            images = chunk['images']
        else:
            message_chunks.append(chunk)

    return message_chunks, images


st.title('Chat RAG')

# Track processed files to avoid re-embedding
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Track skipped files (user clicked cancel) - cleared when uploader changes
if 'skipped_files' not in st.session_state:
    st.session_state.skipped_files = set()

# Track pending file replacements
if 'pending_replace' not in st.session_state:
    st.session_state.pending_replace = None

# Track previous uploaded file names to detect changes
if 'prev_uploaded_names' not in st.session_state:
    st.session_state.prev_uploaded_names = set()

# Get existing documents once
loop = get_or_create_event_loop()
existing_docs = set(loop.run_until_complete(get_indexed_documents()))


def process_file(file_path: Path, file_name: str, replace: bool = False) -> None:
    """Process and embed a file, optionally replacing existing."""
    if replace:
        loop = get_or_create_event_loop()
        loop.run_until_complete(delete_document(file_name))

    loop = get_or_create_event_loop()
    loop.run_until_complete(process_and_embed_document(file_path))
    st.session_state.processed_files.add(file_name)


# File upload section in sidebar
with st.sidebar:
    st.header('Upload Files')
    uploaded_files = st.file_uploader(
        'Upload documents to add to the knowledge base',
        type=['txt', 'pdf', 'md', 'json', 'csv', 'docx', 'pptx', 'xlsx', 'html', 'tsv'],
        accept_multiple_files=True,
        key='file_uploader',
    )

    # Get current uploaded file names
    current_names = {f.name for f in uploaded_files} if uploaded_files else set()

    # Clear skipped files that are no longer in the uploader
    # This allows re-uploading a previously skipped file
    removed_files = st.session_state.prev_uploaded_names - current_names
    for removed in removed_files:
        st.session_state.skipped_files.discard(removed)
    st.session_state.prev_uploaded_names = current_names

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip already processed or skipped in this session
            if uploaded_file.name in st.session_state.processed_files:
                continue
            if uploaded_file.name in st.session_state.skipped_files:
                continue

            file_path = DATA_DIR / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Check if document already exists in DB
            if uploaded_file.name in existing_docs:
                st.session_state.pending_replace = {
                    'name': uploaded_file.name,
                    'path': file_path,
                }
            else:
                with st.spinner(f'Processing {uploaded_file.name}...'):
                    try:
                        process_file(file_path, uploaded_file.name)
                        st.success(
                            f'Processed and embedded: {uploaded_file.name}'
                        )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        st.error(f'Error processing {uploaded_file.name}: {e}')

    # Process file replacement if confirmed
    if st.session_state.get('confirmed_replace'):
        replace_info = st.session_state.confirmed_replace
        name = replace_info['name']
        path = replace_info['path']
        with st.spinner(f'Replacing {name}...'):
            try:
                process_file(path, name, replace=True)
                st.success(f'Replaced and embedded: {name}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f'Error replacing {name}: {e}')
        st.session_state.confirmed_replace = None

    # Replace confirmation dialog
    if st.session_state.pending_replace:
        pending = st.session_state.pending_replace
        name = pending['name']
        path = pending['path']

        @st.dialog(f'Replace "{name}"?')
        def confirm_replace() -> None:
            st.write(
                f'**{name}** already exists in the knowledge base. '
                'Do you want to replace it?'
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Cancel', use_container_width=True):
                    st.session_state.pending_replace = None
                    st.session_state.skipped_files.add(name)
                    st.rerun()
            with col2:
                if st.button('Replace', type='primary', use_container_width=True):
                    st.session_state.confirmed_replace = {
                        'name': name,
                        'path': path,
                    }
                    st.session_state.pending_replace = None
                    st.rerun()

        confirm_replace()

    # Show indexed documents
    st.divider()
    st.header('Indexed Documents')

    # Initialize selected docs in session state
    if 'selected_docs' not in st.session_state:
        st.session_state.selected_docs = []

    try:
        loop = get_or_create_event_loop()
        documents = loop.run_until_complete(get_indexed_documents())
        if documents:
            # Document filter multiselect
            selected = st.multiselect(
                'Filter RAG to specific documents:',
                options=documents,
                default=st.session_state.selected_docs,
                placeholder='All documents (no filter)',
            )
            st.session_state.selected_docs = selected

            if selected:
                st.caption(f'RAG limited to {len(selected)} document(s)')
            else:
                st.caption('RAG searching all documents')

            st.divider()

            for doc in documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f'📄 {doc}')
                with col2:
                    if st.button('🗑️', key=f'del_{doc}', help=f'Delete {doc}'):
                        st.session_state.doc_to_delete = doc

            # Confirmation dialog
            if 'doc_to_delete' in st.session_state:
                doc = st.session_state.doc_to_delete

                @st.dialog(f'Delete "{doc}"?')
                def confirm_delete() -> None:
                    st.write(
                        f'Are you sure you want to delete **{doc}**? '
                        'This will remove all embeddings for this document.'
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button('Cancel', use_container_width=True):
                            del st.session_state.doc_to_delete
                            st.rerun()
                    with col2:
                        if st.button(
                            'Delete', type='primary', use_container_width=True
                        ):
                            loop = get_or_create_event_loop()
                            loop.run_until_complete(delete_document(doc))
                            st.session_state.processed_files.discard(doc)
                            del st.session_state.doc_to_delete
                            st.rerun()

                confirm_delete()
        else:
            st.info('No documents indexed yet.')
    except Exception as e:
        st.error(f'Could not load documents: {e}')

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize conversation memory
if 'memory' not in st.session_state:
    st.session_state.memory = None

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

        # Render any stored images for this message
        if 'images' in message and message['images']:
            for i, img_base64 in enumerate(message['images']):
                try:
                    img_bytes = base64.b64decode(img_base64)
                    st.image(img_bytes, caption=f"Generated Plot {i + 1}")
                except Exception:
                    pass

if prompt := st.chat_input('What is up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    # Get document count before response (to detect new docs added by agent)
    loop = get_or_create_event_loop()
    docs_before = set(loop.run_until_complete(get_indexed_documents()))

    # Store images from agent - use a mutable container for closure access
    image_container: dict[str, list[str]] = {"images": []}

    with st.chat_message('assistant'):

        def stream_sync() -> Generator[str, None, None]:
            """
            Transform the AsyncGenerator into a Generator, as Streamlit
            `.write_stream` doesn't support async.
            """
            assert prompt, "Variable 'prompt' is empty or not set!"
            loop = get_or_create_event_loop()
            # Pass selected documents filter (None if empty list)
            selected = st.session_state.get('selected_docs') or None
            # Pass message history for context (exclude current message)
            history = st.session_state.messages[:-1] if st.session_state.messages else []
            # Pass conversation memory
            memory = st.session_state.get('memory')
            messages, images = loop.run_until_complete(
                stream_response(prompt, selected, history, memory)
            )
            image_container["images"] = images

            for message in messages:
                yield message
                time.sleep(0.05)

        response = st.write_stream(stream_sync)

        # Render any generated images from code execution
        for i, img_base64 in enumerate(image_container["images"]):
            try:
                img_bytes = base64.b64decode(img_base64)
                st.image(img_bytes, caption=f"Generated Plot {i + 1}")
            except Exception:
                pass

    generated_images = image_container["images"]
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'images': generated_images,
    })

    # Update memory with the new exchange (in background)
    loop = get_or_create_event_loop()
    try:
        new_memory = loop.run_until_complete(
            update_memory(st.session_state.memory, prompt, response)
        )
        st.session_state.memory = new_memory
    except Exception:
        pass  # Don't fail the chat if memory update fails

    # Refresh sidebar if new documents were added to KB by the agent
    docs_after = set(loop.run_until_complete(get_indexed_documents()))
    if docs_after != docs_before:
        st.rerun()
