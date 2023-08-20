import re
import pandas as pd
import requests
from flask import Flask, jsonify, request
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)

def add_space_to_korean_words(text):
    pattern = re.compile(r'(?<![ㄱ-ㅎㅏ-ㅣ가-힣])((?!도|시|군|구|읍|면|로|길)[ㄱ-ㅎㅏ-ㅣ가-힣]+)')
    result = re.sub(pattern, r' \1', text)
    return result

def add_space_to_uppercase_letters(text):
    pattern = re.compile(r'(?<=[a-zA-Z\d가-힣])(?=[A-Z])')
    result = re.sub(pattern, ' ', text)
    return result

def add_space_to_numbers(text):
    pattern = re.compile(r'(?<!-)(?<!\d)(\d+)')
    result = re.sub(pattern, r' \1', text)
    return result

def remove_commas(text):
    result = text.replace(',', '')
    return result


def process_address_patterns(address):
    pattern1 = r"\b[\w-]*-do\b|\b[\w-]*도\b"
    pattern2 = r"\b[\w-]*-si\b|\b[\w-]*시\b|\bSeoul\b|\b서울\b|\bBusan\b|\b부산\b|\bGwangju\b|\b광주\b|\bDaegu\b|\b대구\b|\bDaejeon\b|\b대전\b|\bUlsan\b|\b울산\b|\bIncheon\b|\b인천\b"
    pattern3 = r"\b[\w-]*-gu\b|\b[\w-]*구\b|\b[\w-]*gu\b"
    pattern4 = r"\b[\w-]*-gun\b|\b[\w-]*군\b"
    pattern5 = r"\b[\w-]*-eup\b|\b[\w-]*읍\b"
    pattern6 = r"\b[\w-]*-myeon\b|\b[\w-]*(?<!으)면\b"
    pattern7 = r"\b[\w-]*로\b|\b[\w-]*-daero\b|\b[\w-]*-ro\b|\b\w+\s*Station-ro\b|\b\w+\s*Ring-ro\b"
    pattern8 = r"\b[\w-]*-gil\b|\b[\w-]*길\b"
    pattern9 = r"(?<!\S)(G|B|GF|BF|G/F|underground|B/F|지하|(?<=\s)B(?=\,))(?!\S)" 
    pattern10 = r"(?<!\S)(\d+(?:-\d+)?)(?!\S)"
    
    patterns = [
        (pattern1, lambda match: match.group(0) + " "),
        (pattern2, lambda match: match.group(0) + " "),
        (pattern3, lambda match: match.group(0) + " "),
        (pattern4, lambda match: match.group(0) + " "),
        (pattern5, lambda match: match.group(0) + " "),
        (pattern6, lambda match: match.group(0) + " "),
        (pattern7, lambda match: match.group(0) + " "),
        (pattern8, lambda match: match.group(0) + " "),
        (pattern9, lambda match: re.sub(pattern9, '지하', match.group(0)) + " "),
        (pattern10, lambda match: match.group(0) + " ")
    ]
    
    result = ""
    for pattern, replacement_func in patterns:
        match = re.search(pattern, address)
        if match:
            result += replacement_func(match)
    
    return result.strip()


def convert_hybrid_words(text):
    # 정규표현식을 사용하여 한영혼용단어를 찾습니다.
    pattern1 = r'([가-힣]+)-do'
    pattern2 = r'([가-힣]+)-si'
    pattern3 = r'([가-힣]+)-gu'
    pattern4 = r'([가-힣]+)-gun'
    pattern5 = r'([가-힣0-9]+)-ro'
    pattern6 = r'([가-힣0-9]+)-gil'
    
    pattern7 = r'([a-zA-Z]+)도'
    pattern8 = r'([a-zA-Z]+)시'
    pattern9 = r'([a-zA-Z]+)구'
    pattern10 = r'([a-zA-Z]+)군'
    pattern11 = r'([a-zA-Z]+)로'
    pattern12 = r'([a-zA-Z]+)길'
    
    matches1 = re.findall(pattern1, text)
    matches2 = re.findall(pattern2, text)
    matches3 = re.findall(pattern3, text)
    matches4 = re.findall(pattern4, text)
    matches5 = re.findall(pattern5, text)
    matches6 = re.findall(pattern6, text)
    matches7 = re.findall(pattern7, text)
    matches8 = re.findall(pattern8, text)
    matches9 = re.findall(pattern9, text)
    matches10 = re.findall(pattern10, text)
    matches11 = re.findall(pattern11, text)
    matches12 = re.findall(pattern12, text)
    
    for match in matches1:
        converted_word = match + '도'
        text = text.replace(match + '-do', converted_word)
    for match in matches2:
        converted_word = match + '시'
        text = text.replace(match + '-si', converted_word)
    for match in matches3:
        converted_word = match + '구'
        text = text.replace(match + '-gu', converted_word)
    for match in matches4:
        converted_word = match + '군'
        text = text.replace(match + '-gun', converted_word)
    for match in matches5:
        converted_word = match + '로'
        text = text.replace(match + '-ro', converted_word)
    for match in matches6:
        converted_word = match + '길'
        text = text.replace(match + '-gil', converted_word)
    for match in matches7:
        converted_word = match + '-do'
        text = text.replace(match + '도', converted_word)
    for match in matches8:
        converted_word = match + '-si'
        text = text.replace(match + '시', converted_word)
    for match in matches9:
        converted_word = match + '-gu'
        text = text.replace(match + '구', converted_word)
    for match in matches10:
        converted_word = match + '-gun'
        text = text.replace(match + '군', converted_word)
    for match in matches11:
        converted_word = match + '-ro'
        text = text.replace(match + '로', converted_word)
    for match in matches12:
        converted_word = match + '-gil'
        text = text.replace(match + '길', converted_word)
    
    text = text.replace('-로', '-ro')
    text = text.replace('beon-gil', '번길')
    text = text.replace(' Ring-ro', 'sunhwan-ro')
    text = text.replace(' Station-ro', 'Station-ro')
    
    return text



# Load data from the other Excel file (contains the mapping)
mapping_file = 'data.csv'
mapping_df = pd.read_csv(mapping_file)

# Create a dictionary mapping English words to Korean words
mapping_dict = dict(zip(mapping_df['로마자표기'], mapping_df['한글']))

# 함수 내 영어 단어를 한글로 변환하는 부분
def replace_english_with_korean(sentence):
    def replace_word(match):
        word = match.group(0)
        return mapping_dict.get(word, word)

    return re.sub(r'\b[A-Za-z-]+\b', replace_word, sentence)

def remove_underground_numbers(column_value):
    if re.match(r'^지하\s?\d+$', column_value):
        return '답 없음 나와야 함'
    return column_value



# TrieNode와 Trie 클래스
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, value):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = value

    def search_closest_word(self, word, max_distance):
        current_row = list(range(len(word) + 1))
        results = []

        for char in self.root.children:
            self._search_recursive(self.root.children[char], char, word, current_row, results, max_distance)

        if not results:
            return None
        results.sort(key=lambda x: x[0])
        return results[0][1]

    def _search_recursive(self, node, char, word, previous_row, results, max_distance):
        columns = len(word) + 1
        current_row = [previous_row[0] + 1]

        for col in range(1, columns):
            insert_cost = current_row[col - 1] + 1
            delete_cost = previous_row[col] + 1
            replace_cost = previous_row[col - 1]

            if word[col - 1] != char:
                replace_cost += 1

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        if current_row[-1] <= max_distance and node.is_end_of_word:
            results.append((current_row[-1], node.value))

        if min(current_row) <= max_distance:
            for child_char in node.children:
                self._search_recursive(node.children[child_char], child_char, word, current_row, results, max_distance)

# 단어가 영어인지 확인하는 함수
def is_english(word):
    return bool(re.match('^[a-zA-Z\s-]+$', word))

# 단어를 교정하고 번역하는 함수
def correct_and_translate(input_word, eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie):
    if is_english(input_word):
        max_distance = 3
        trie = eng_trie
        mapping_dict = eng_mapping_dict
    else:
        max_distance = 1
        trie = kor_trie
        mapping_dict = kor_mapping_dict

    corrected_word = trie.search_closest_word(input_word, max_distance)
    translated_word = mapping_dict.get(corrected_word, corrected_word)
    return translated_word

def process_address(input_address, eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie):
    elements = input_address.split()
    corrected_elements = []

    for element in elements:
        if (element.isdigit() or element == '지하' or 
            re.match(r'^\d+번길$|^\d+로$|^\d+길$', element) or
            re.match(r'^[\d-]+$', element)):
            corrected_element = element
        else:
            corrected_element = correct_and_translate(element, eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie)
            if not corrected_element:
                corrected_element = element

        corrected_elements.append(corrected_element)

    return ' '.join(corrected_elements)

def create_mapping_tries(mapping_df):
    eng_mapping_dict = {}
    kor_mapping_dict = {}
    eng_trie = Trie()
    kor_trie = Trie()

    for index, word in enumerate(mapping_df['로마자표기']):
        if is_english(word):
            eng_trie.insert(word, mapping_df['한글'][index])
            eng_mapping_dict[word] = mapping_df['한글'][index]
        else:
            kor_trie.insert(word, mapping_df['한글'][index])
            kor_mapping_dict[word] = mapping_df['한글'][index]

    return eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie

# Create the mapping tries
eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie = create_mapping_tries(mapping_df)
    
async def process_address_async(address, eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie):
    formatted_address = address
    formatted_address = add_space_to_korean_words(formatted_address)
    formatted_address = add_space_to_uppercase_letters(formatted_address)
    formatted_address = add_space_to_numbers(formatted_address)
    formatted_address = remove_commas(formatted_address)
    formatted_address = process_address_patterns(formatted_address)
    result = convert_hybrid_words(formatted_address.strip())
    result = replace_english_with_korean(result.strip())
    result = remove_underground_numbers(result.strip())
    result = process_address(result.strip(), eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie)
    return result

@app.route('/search', methods=['POST'])
async def search():
    try:
        if request.is_json:
            request_data = request.get_json()
        else:
            request_data = {'requestList': [{'seq': '000001', 'requestAddress': request.data.decode('utf-8')}]}
        
        request_list = request_data.get('requestList', [])
        results = []

        async def process_request(req):
            seq = req.get('seq')
            address = req.get('requestAddress')
            result_address = await asyncio.to_thread(perform_address_search, address)
            
            if len(result_address) == 0:
                results.append({'seq': seq, 'resultAddress': 'F'})
            elif len(result_address) >= 1:
                processed_address = await process_address_async(address, eng_mapping_dict, kor_mapping_dict, eng_trie, kor_trie)
                results.append({'seq': seq, 'resultAddress': processed_address})

        asyncio.run(asyncio.gather(*[process_request(req) for req in request_list]))

        response_data = {'HEADER': {'RESULT_CODE': 'S', 'RESULT_MSG': 'Success'}, 'BODY': results}
        return jsonify(response_data)
    except Exception as e:
        response_data = {'HEADER': {'RESULT_CODE': 'F', 'RESULT_MSG': str(e)}}
        return jsonify(response_data)

# Function to create a session with custom retry and timeout settings
def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session



# Function to perform address search using the session with retries
def perform_address_search(search_data):
    api_key = 'devU01TX0FVVEgyMDIzMDcyODE1MzkzNzExMzk3MzA='
    base_url = 'http://www.juso.go.kr/addrlink/addrLinkApi.do'

    payload = {
        'confmKey': api_key,
        'currentPage': '1',
        'countPerPage': '10',
        'resultType': 'json',
        'keyword': search_data,
    }

    session = create_session()

    try:
        response = session.get(base_url, params=payload, timeout=1200)
        if response.status_code == 200:
            search_result = response.json()
            if 'results' in search_result and 'juso' in search_result['results']:
                result_data = search_result['results']['juso']
                if result_data:
                    # Extract and return the road addresses from the API response
                    return [result.get('roadAddr', '') for result in result_data]
    except Exception as e:
        print("An error occurred:", e)

    return []



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)  



  
