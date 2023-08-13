import re
from pandas import pandas as pd
import requests
from flask import Flask, jsonify, request
from gunicorn.app.wsgiapp import run


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
# 함수 내 영어 단어를 한글로 변환하는 부분
def replace_english_with_korean(sentence):
    def replace_word(match):
        word = match.group(0)
        return mapping_dict.get(word, word)

    return re.sub(r'\b[A-Za-z-]+\b', replace_word, sentence)

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min(distances[index1], distances[index1 + 1], new_distances[-1]))
        distances = new_distances

    return distances[-1]

def correct_typo(input_word, valid_words):
    min_distance = float('inf')
    corrected_word = None

    for word in valid_words:
        distance = levenshtein_distance(input_word, word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = word

    return corrected_word

def correct_and_translate(input_word, valid_words, mapping_dict):
    corrected_word = correct_typo(input_word, valid_words)
    translated_word = mapping_dict.get(corrected_word, corrected_word)
    return translated_word


def correct_and_translate_address(address, mapping_df):
    elements = address.split()
    corrected_elements = []

    for element in elements:
        if (element.isdigit() or element == '지하' or 
            re.match(r'^\d+번길$|^\d+로$|^\d+길$', element) or
            re.match(r'^[\d-]+$', element)):
            corrected_element = element
        elif re.match(r'^[a-zA-Z\s-]+$', element):
            corrected_element = correct_and_translate(element, mapping_df['로마자표기'], mapping_dict)
        else:
            # Check if the Korean word exists in the mapping data, if not, find the closest correction
            if element not in mapping_df['한글'].tolist():
                corrected_element = correct_typo(element, mapping_df['한글'])
            else:
                corrected_element = element

        corrected_elements.append(corrected_element)

    corrected_address = ' '.join(corrected_elements)
    return corrected_address

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

    response = requests.get(base_url, params=payload)
    
    

    if response.status_code == 200:
        search_result = response.json()
        if 'results' in search_result and 'juso' in search_result['results']:
            result_data = search_result['results']['juso']
            if result_data:
                return [result.get('roadAddr', '') for result in result_data]

    return ['F']


# Load data from the Excel file (contains the mapping)
mapping_file = 'data.xlsx'
mapping_df = pd.read_excel(mapping_file)
mapping_dict = dict(zip(mapping_df['로마자표기'], mapping_df['한글']))

# 주소 전처리 및 검색 요청 함수 정의
@app.route('/send_request', methods=['POST'])
def send_request():
    try:
        if request.is_json:
            request_data = request.get_json()
        else:
            request_data = {'requestList': [{'seq': '000001', 'requestAddress': request.data.decode('utf-8')}]}
        
        request_list = request_data.get('requestList', [])

        results = []

        for req in request_list:
            seq = req.get('seq')
            address = req.get('requestAddress')

            formatted_address = address
            print("Original Address:", formatted_address)

            formatted_address = add_space_to_korean_words(formatted_address)
            print("After add_space_to_korean_words:", formatted_address)

            formatted_address = add_space_to_uppercase_letters(formatted_address)
            print("After add_space_to_uppercase_letters:", formatted_address)

            formatted_address = add_space_to_numbers(formatted_address)
            print("After add_space_to_numbers:", formatted_address)

            formatted_address = remove_commas(formatted_address)
            print("After remove_commas:", formatted_address)

            # 패턴 매치 수행
            formatted_address = process_address_patterns(formatted_address)
            print("After process_address_patterns:", formatted_address)

            result = convert_hybrid_words(formatted_address.strip())
            print("After convert_hybrid_words:", result)

            result = replace_english_with_korean(result.strip())  # 영어 단어 한글 변환 적용
            print("After replace_english_with_korean:", result)
            
            result = correct_and_translate_address(result, mapping_df)
            print("correct_and_translate_address:", result)


            
            # 주소 검색 결과 가져오기
            result_address = perform_address_search(result)

            if len(result_address) == 1:
                results.append({'seq': seq, 'resultAddress': result_address[0]})
            else:
                results.append({'seq': seq, 'resultAddress': 'F'})

        response_data = {'HEADER': {'RESULT_CODE': 'S', 'RESULT_MSG': 'Success'}, 'BODY': results}
        return jsonify(response_data)
    except Exception as e:
        response_data = {'HEADER': {'RESULT_CODE': 'F', 'RESULT_MSG': str(e)}}
        return jsonify(response_data)


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

    response = requests.get(base_url, params=payload)

    if response.status_code == 200:
        search_result = response.json()
        print("Address API Response:", search_result)  # 추가된 출력문
        if 'results' in search_result and 'juso' in search_result['results']:
            result_data = search_result['results']['juso']
            if result_data:
                # Extract and return the road addresses from the API response
                return [result.get('roadAddr', '') for result in result_data]

    return ['F']



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
