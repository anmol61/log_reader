import unittest
from unittest.mock import patch, MagicMock
import os
from main import get_log_file, doc_splitter, get_vector_store, get_conversational_chain, summary_docs, search_similarity, main
 
class TestMain(unittest.TestCase):
 
    def setUp(self):
        self.log_doc = '/path/to/log/doc'
        self.docs = ['doc1', 'doc2', 'doc3']
        self.splitted_docs = ['split_doc1', 'split_doc2', 'split_doc3']
        self.user_input = 'test input'
        self.summary = 'summary of docs'
        self.response = {'output_text': 'test response'}
 
    @patch('main.TextLoader')
    def test_get_log_file(self, mock_text_loader):
        mock_text_loader.return_value.load.return_value = self.docs
        result = get_log_file(self.log_doc)
        self.assertEqual(result, self.docs)
 
    @patch('main.RecursiveCharacterTextSplitter')
    def test_doc_splitter(self, mock_text_splitter):
        mock_text_splitter.return_value.split_documents.return_value = self.splitted_docs
        result = doc_splitter(self.docs)
        self.assertEqual(result, self.splitted_docs)
 
    @patch('main.OpenAIEmbeddings')
    @patch('main.FAISS')
    def test_get_vector_store(self, mock_faiss, mock_embeddings):
        get_vector_store(self.splitted_docs)
        mock_faiss.from_documents.assert_called_once_with(self.splitted_docs, embedding=mock_embeddings.return_value)
 
    @patch('main.OpenAI')
    @patch('main.PromptTemplate')
    @patch('main.load_qa_chain')
    def test_get_conversational_chain(self, mock_load_qa_chain, mock_prompt_template, mock_openai):
        get_conversational_chain()
        mock_load_qa_chain.assert_called_once()
 
    @patch('main.OpenAI')
    @patch('main.PromptTemplate')
    @patch('main.load_qa_chain')
    def test_summary_docs(self, mock_load_qa_chain, mock_prompt_template, mock_openai):
        mock_load_qa_chain.return_value.return_value = {'output_text': self.summary}
        result = summary_docs(self.docs)
        self.assertEqual(result, [self.summary] * 8)
 
    @patch('main.OpenAIEmbeddings')
    @patch('main.FAISS')
    @patch('main.Document')
    @patch('main.RecursiveCharacterTextSplitter')
    def test_search_similarity(self, mock_text_splitter, mock_document, mock_faiss, mock_embeddings):
        mock_faiss.return_value.similarity_search.return_value = [mock_document]
        mock_document.page_content = self.summary
        mock_document.metadata = {'source': 'source'}
        search_similarity(self.user_input)
        mock_faiss.load_local.assert_called_once()
 
    @patch('main.st')
    @patch('main.search_similarity')
    def test_main(self, mock_search_similarity, mock_st):
        mock_st.text_input.return_value = 'test input'
        main()
        mock_st.set_page_config.assert_called_once_with(page_title="Log Analyzer")
        mock_st.header.assert_called_once_with("Chat with the bot to get insights from the Log Files")
        mock_search_similarity.assert_called_once_with(user_input='test input')
 
if __name__ == '__main__':
    unittest.main()