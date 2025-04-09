#!/usr/bin/env python
# coding: utf-8

# # Acknowledgement & Revision Notice
# 
# This lab session material has been adapted and modified based on previous works by multiple contributors. The original tutorial was written by [Justin Johnson](https://web.eecs.umich.edu/~justincj/) for CS231n and later adapted as a Jupyter notebook for CS228 by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335). The current version with Python 3 support was further adapted by Kevin Zakka for the Spring 2020 edition of [CS231n](http://cs231n.stanford.edu/).

# # Jupyter Notebook 사용하기
# 
# Jupyter notebook은 여러개의 <b>셀(cell)</b>로 구성되어 있으며 각각의 cell 은 <b>Python 코드</b> 또는 <b>설명(텍스트)</b>을 담을 수 있습니다.
# 
# 셀은 크게 두 가지 유형으로 나뉩니다:
# - `Code` cell : Python 코드를 작성하고 실행할 수 있습니다.
# - `Markdown` cell: 설명을 작성할 수 있으며, Markdown 문법을 활용하여 텍스트를 꾸밀 수 있습니다.
# 
# 현재 보고 있는 이 cell은 `Markdown` cell 입니다
# 
# ### 셀 편집 및 실행 방법
# 
# - `Markdown` 셀 수정: 더블클릭하면 편집 모드로 전환됩니다.
# - Cell 실행: 원하는 셀을 선택한 후 `Shift + Enter`를 입력합니다.
#   - `Code` 셀인 경우 → 코드가 실행되며, 실행 결과가 셀 아래에 출력됩니다.
#   - `Markdown` 셀인 경우 → Markdown 문법이 적용된 형태로 렌더링됩니다.
# 
# 
# 아래는 `Code` cell입니다.

# Cell간에는 전역 변수들이 공유됩니다.

# ### Keyboard Shortcuts (단축키)
# 
# * `esc`: 셀(Cell) 편집 종료 및 **명령 모드**로 전환
# * 셀 이동: 화살표 (`↑`, `↓`)를 사용하여 cell간에 이동할 수 있다.
# * 셀 추가:
#   - `b`: 현재 cell 아래에 새로운 cell 삽입
#   - `a`: 현재 cell 위에 새로운 cell 삽입
# * 셀 삭제: `dd` (연속 두 번 입력)
# * 셀 타입 변경:
#   - `m` : cell을 `Markdown`으로 변경 (명령 모드 `Esc` 상태여야 함)
#   - `y` : cell을 `Code`으로 변경 (명령 모드 `Esc` 상태여야 함)
# 
# ### 커널(Kernel) 관리
# 
# Jupyter Notebook에서 **커널(Kernel)** 은 코드 실행을 담당하는 프로세스로, 실행 중 오류가 발생하거나 모든 변수를 초기화하려면 커널을 재시작해야 합니다.
# 
# - **커널 재시작**: `Kernel -> Restart` 을 통해 파이썬 커널을 재시작할 수 있습니다 (코드 실행결과는 그대로 유지됨)
# - **커널 재시작 및 초기화**: `Kernel -> Restart & Clear Output` 을 통해 커널을 재시작하고 모든 코드 실행결과를 삭제할 수 있습니다
# - **전체 셀 실행**: `Run -> Run All Cell` 를 클릭하면 노트북의 모든 셀이 위에서 아래로 순차적으로 실행됩니다.
# 
# Jupyter Notebook은 **위에서 아래로 실행하는 것이 원칙**입니다.  
# 커널을 재시작하면 모든 변수와 함수 정의가 사라지므로, **모든 Cell을 처음부터 다시 실행해야 합니다**.  
# 특정 셀을 건너뛰거나 순서를 바꾸어 실행하면 **변수나 함수가 정의되지 않거나 예상치 못한 오류**가 발생할 수 있습니다.
# 
# 커널을 재시작 한 뒤 위의 `y = 2 * x ` 셀을 실행하여 무슨 일이 일어나는지 확인해보세요.

# # Python Essentials
# Python은 **범용 프로그래밍 언어**(General-purpose programming language)로 널리 사용되며, 특히, `NumPy`, `SciPy`, `Matplotlib` 등의 라이브러리와 함께 사용하면 데이터 분석, 수치 계산, 시각화를 포함한 다양한 작업을 매우 효과적으로 수행할 수 있습니다.
# 
# 이번 실습에서는 다음과 같은 핵심 개념을 다룹니다:
# 
# * 파이썬 기본 문법 및 데이터 타입 (Containers, Lists, Dictionaries, Sets, Tuples)
# * 함수와 클래스와 객체 지향 프로그래밍
# * 매직 메서드
# * 컨텍스트 관리자
# 
# ## 파이썬의 특징
# Python은 고수준(High-level), 동적 타이핑(Dynamically typed) 을 지원하는 다중 패러다임(Multiparadigm) 프로그래밍 언어입니다.  
# 즉, Python은 절차적(Procedural), 객체 지향(Object-oriented), 함수형(Functional) 프로그래밍을 모두 지원하며, 코드가 간결하면서도 강력한 표현력을 가집니다.
# 
# Python 코드는 종종 **의사 코드(Pseudocode)** 와 유사하다고 평가될 정도로 직관적이며, **적은 코드 라인**으로도 복잡한 개념을 쉽게 표현할 수 있을 뿐만 아니라 **가독성** 또한 뛰어납니다.
# 
# 예를 들어, 아래는 **Python을 이용한 퀵소트(Quicksort) 알고리즘**의 구현입니다:

# ### Python versions
# Python은 크게 **2.x**와 **3.x** 버전으로 나뉩니다.  
# Python 3.0이 출시되면서 **기존 2.x 코드와의 호환성이 중단되는 큰 변화**가 있었습니다.  
# 
# 현재 Python 2.x는 2020년 1월부로 공식 지원이 종료되었으며,  
# 대부분의 최신 라이브러리는 Python 3.x에서만 정상적으로 동작합니다.
# 
# 따라서, 최신 기능과 안정성을 고려하여  
# 딥러닝 및 머신러닝을 위한 작업에서는 Python 3.8 이상 사용을 권장합니다.

# ## Basic data types
# ### 숫자형 (Numeric)
# 정수(Integers), 실수(floats), 복소수 (Complex number)에 대한 사칙연산이 잘 정의되어 있다.

# ### 불리언(Booleans)
# 
# **불리언(Boolean)** 타입은 `True` 또는 `False` 값을 가지며, 주로 **조건문, 논리 연산, 비교 연산**에서 사용됩니다.
# 
# Python에서는 **불리언 연산자**로 `and`, `or`, `not`을 사용하며, 다른 프로그래밍 언어에서 흔히 쓰이는 기호(`&&`, `||`, `!`) 대신 직관적인 영어 단어를 사용합니다.
# 

# 비교 연산자는 불리언(Boolean) 타입의 값을 반환합니다.
# 
# 또한, 불리언 타입은 조건문에서 프로그램의 흐름을 제어하는 핵심 요소로 자주 사용됩니다.

# ### 문자열(Strings)

# `f-string`을 사용하면 다른 데이터 타입과 문자열을 쉽게 결합할 수 있습니다.

# 문자열을 다루는 다양한 메서드들이 잘 정의되어 있다.

# ## 컨테이너 (Containers)
# 컨테이너는 자료를 담는 그릇이라는 뜻으로, 파이썬에는, lists, dictionaries, sets, tuples등이 정의되어 있습니다.
# 
# ### 리스트 (Lists)
# 리스트(List)는 **순서(Ordered)를 가지며, 중복을 허용하는 객체의 집합**입니다.  
# 또한, **크기가 동적으로 변경 가능**하며, 서로 다른 타입의 데이터도 저장할 수 있습니다.

# #### 슬라이싱(Slicing)
# Python에서는 개별 요소를 **인덱싱(Indexing)** 으로 접근할 수 있지만,  
# 여러 개의 요소를 한 번에 가져오려면 **슬라이싱(Slicing)** 을 사용합니다.  

# #### 루프 (Loops)
# 리스트와 같이 순회가능한(iterable)객체는 `for` 문을을 이용하여 순회할 수 있습니다

# 인덱스(index)와 값(value)를 함께 가져오고 싶다면 enumerate() 함수를 사용할 수 있습니다.

# #### 리스트 컴프리헨션(List comprehensions)
# 아래와 같이 리스트를 순회하며 거듭제곱을 하는 코드를 생각해보자.

# 리스트 컴프리헨션을 사용하면 이 코드를 훨씬 가독성있고 간단하게 만들 수 있습니다.

# 리스트 컴프리헨션은 조건문을 함께 이용할 수도 있습니다.

# ### 딕셔너리 (Dictionaries)
# 딕셔너리는 (key, value) 쌍으로 이루어진 자료형으로,
# 리스트와 같이 순차적으로(sequential)으로 요소를 저장하거나 가져오는것이 아니라,
# 키(key)를 이용하여 원하는 값(value) 빠르게 얻어올 수 있습니다.
# 
# 또한, 딕셔너리는 **해시(Hash) 기반의 자료구조**를 사용하여 **빠른 조회 성능**을 제공합니다.

# 딕셔너리도 이터러블(iterable)이다

# (key, value) 쌍에 접근하고 싶으면 ``items``를 사용한다

# #### Dictionary Comprehension
# 딕셔너리도 리스트와 마찬가지로 딕셔너리 컴프리헨션(Dictionary Comprehension)을 사용하여 간결하게 생성할 수 있습니다.
# 

# ### 집합 (Sets)
# 
# 집합(Set)은 순서가 없고(unordered) 중복을 허용하지 않는 요소들의 모음입니다.  

# 집합을 이용하여 수학적 집합 연산(합집합, 교집합, 차집합 등)을 손쉽게 수행할 수 있습니다.

# 집합은 중복을 자동으로 제거하기 때문에 리스트에서 중복 요소를 제거할 때 유용합니다.

# 집합도 iterable입니다. 하지만 집합에는 순서가 없어서 순서를 예측할 수 없습니다.

# #### Set comprehensions
# 리스트, 딕셔너리와 동일하게 Set comprehensions을 이용할 수 있습니다.

# ### 튜플(Tuples)
# 튜플은 순서가 있는(Ordered) 불변하는(immutable) 값들의 리스트입니다.
# - 튜플은 리스트와 유사하지만 **한 번 생성되면 수정(변경)할 수 없는 특징**을 가집니다.
# - 또한 튜플은 리스트와 달리 딕셔너리의 키(Key)로 사용되거나, 집합(Set)의 요소로 포함될 수 있습니다
# - 반면, 리스트(List)는 **변경 가능(Mutable)한 특성** 때문에 딕셔너리의 키나 집합의 요소로 사용할 수 없습니다.

# ### Mutable과 Immutable
# 
# 파이썬의 모든것은 객체(Object)이며, 이 객체들은 값이 변경가능한 mutable 객체와 변경할 수 없는 immutable 객체로 나뉩니다.
# 
# #### Immutable : 한 번 생성되면 값을 변경할 수 없는 객체
#   - 숫자 (number)
#   - 문자열 (string)
#   - 튜플 (tuple)

# #### Mutable: 값을 직접 수정할 수 있는 객체
#   - 리스트 (list)
#   - 집합 (set)
#   - 딕셔너리 (dictionary)

# <mark>(주의)</mark> mutable 객체는 그 안에 담겨 있는 값이 다른 함수나 연산에 의해 변경될 수 있으니 주의해야 합니다.

# Mutable 객체가 의도치 않게 변경되는 것을 방지하려면 Immutable 객체를 사용하거나, `copy()` 를 활용하여 복사본을 만들어 사용해야 합니다.
# 
# 
# `copy`함수는 얕은 복사 (shallow copy)를 수행하며, 외부 객체는 복사되지만 그 내부의 객체는 원본과 공유됩니다.

# 하지만 copy()는 내부 중첩 리스트까지 복사하지 않습니다.
# 
# 즉, 내부 리스트가 수정되면 원본도 영향을 받을 수 있습니다.

# `copy.deepcopy()`를 사용하여 깊은 복사(Deep Copy) 를 수행하면 내부 객체까지 완전히 새로운 복사본이 생성되므로,
# 원본과 독립적으로 동작합니다.

# ## 함수(Functions)
# 
# 파이썬 함수는 `def` 키워드를 이용해 정의한다.
# 
# 함수 호출에는 괄호 `()` 연산자를 사용한다.
# 
# (참고) **"Parameter(매개변수)"** 와 **"Argument(인자)"** 는 서로 다른 개념입니다.
# 
# | 용어 | 설명 |
# |------|----------------|
# | Parameter (매개변수) | 함수 정의 시 선언되는 변수 (값을 받을 자리) |
# | Argument (인자) | 함수 호출 시 전달되는 실제 값 |

# 디폴트 매개변수(Default Parameter)를 이용해 매개변수의 기본값을 설정하면,
# 함수를 호출할 때 해당 인자를 생략할 수 있습니다

# ### 가변인자
# #### Arbitrary Positional Arguments (`*args`)
# 
# 함수의 인자 개수가 가변적이라면, `*args`를 사용하여 임의의 개수의 위치 인자(positional argument)를 받을 수 있습니다.

# print함수는 사실 가변인자를 이용하여 구현됩니다

# #### Arbitrary Keword Arguments (`**kwargs`)
# 인자의 이름을 같이 전달하고 싶을 경우 키워드 인자(keyword argument)를 사용합니다

# 딕셔너리를 이용해 키워드 가변 인자를 전달할 수도 있다

# `*args`와 `**kwargs` 를 함께 사용할 수도 있다.

# ## 클래스 (Classes)
# Python은 객체 지향 프로그래밍(OOP, Object-Oriented Programming) 을 지원하는 언어이며,  
# **클래스(Class)** 는 **데이터(속성, Attributes)와 동작(메서드, Methods)** 를 하나로 묶어 **객체(Object)** 를 생성하는 템플릿 역할을 합니다.
# 
# 즉, 클래스는 특정한 데이터와 해당 데이터에 수행할 수 있는 동작을 정의하여, 이를 기반으로 여러 개의 객체를 생성하고 편리하게 다룰 수 있도록 합니다.
# 이를 통해 코드의 재사용성, 유지보수성, 확장성이 향상됩니다.

# ## 매직 메서드(Magic Method)와 파이썬 객체
# 
# Python에서는 모든 것이 객체(Object)이며, 클래스에 매직 메서드(Magic Method)를 구현하면, Python 문법과 자연스럽게 연동할 수 있습니다.
# 
# ``` python
# x = 5
# print(type(x))  # <class 'int'>
# print(x + 3)    # 8 (내부적으로 x.__add__(3) 호출)
# ```
# 
# 매직 메서드는 `__init__`, `__str__`, `__contains__` 와 같이 이중 밑줄 (double underscore, `__`)로 시작하고 끝나는 특별한 메서드입니다. 이 메서드를 구현하면 파이썬 기본 연산자(+, in, ==)나 함수, 키워드 동작을 사용자 정의 가능하게 합니다.
# 
# 예를 들어:
# 
# - `__init__()` → 객체가 생성될 때 자동 호출되는 생성자 메서드
# - `__str__()` → print() 함수에서 객체를 문자열로 변환할 때 호출
# - `__eq__()` → == 연산자가 사용될 때 호출
# 
# 
# ### 컨테이너 (Container)
# - 컨테이너는 여러개의 요소를 포함할 수 있는 객체를 지칭합니다.
# - `__contains__()` 매직메서드를 구현하고 있으며, 이를 통해 특정 요소가 포함되어 있는지를 확인할 수 있습니다.
# - `__contains__()` 매직메서드는 ``in`` 키워드가 발견될때 호출됩니다.
# - 예시: `list`, `tuple`, `set`, `dict`

# ### 이터러블 (Iterable, 반복 가능한 객체)
# - `for`루프를 통해 순회할 수 있으며, `iter()`함수를 호출하여 이터레이터(iterator)를 생성할 수 있는 객체.
# - `__iter__()` 또는 `__getitem__()`을 구현하면 Iterable이 됨.
# - 이터레이터(iterator)란 `__next__()` 메서드를 구현한 객체. `next()`를 호출하면 다음 값을 반환함
#   - `StopIteration`은 이터레이터가 더 이상 반환할 값이 없을 때 발생하는 예외
# - ``__iter__``와 ``__next__`` 매직 메서드를 구현하여 사용자 정의 이터러블 객체를 구현할 수 있다.

# For loop의 작동원리는 처음에 `iter()`함수를 호출 한 뒤, ``StopIteration`` 예외가 작성할 때 까지 ``next()``를 반복적으로 호출하는 것이다.

# ### 시퀀스 (Sequence)
# - 요소들이 특정한 순서를 가지고 있는 컨테이너 객체로, `len()` 함수를 통해 총 길이를 알 수 있으며 인덱싱(`[]`)을 통해 개별 요소(element)에 접근할 수 있다.
# - `__getitem__`와 `__len__`을 구현하여 사용자 정의 시퀀스 객체를 구현할 수 있다.
#   - `__getitem__`: 객체가 인덱싱(`[]`) 될 때 호출된다.
#   - `__len__` : `len()` 함수 호출 시 호출되며 객체의 총 element 수를 반환한다.
# - 예시: `list`, `tuple`, `str`

# ### 호출가능한(callable) 객체
# 
# 파이썬에서 함수의 호출은 함수의 이름에 `()`를 붙여주면 된다. 왜 `()`를 붙여주면 함수가 호출될까?
# 
# 이는 클래스(타입)의 객체가 있을 때 `()`를 붙여주면 해당 클래스에 정의된 매직 메소드인 `__call__`이 호출되기 때문입니다.
# 
# 즉, 호출가능한(callable) 객체는 ``__call__`` 메직메서드를 구현한 클래스의 인스턴스를 의미합니다.
# 
# 함수도 사실 `__call__()`을 구현한 <b>객체(Object)</b>이므로, `()`로 호출할 수 있습니다.

# ## 컨텍스트 관리자 (Context Manager)
# 컨텍스트 관리자는 특정 작업의 전후에 자동으로 실행되는 동작을 정의할 때 유용합니다.
# 예를 들어, 파일을 열고 나면 파일 디스크립터 누수를 방지하기 위해 반드시 닫아야 합니다.

# Python의 `with` 문을 사용하면 파일을 자동으로 닫을 수 있어 편리합니다

# with 문은 컨텍스트 관리자로 진입하게 하는 역할을 하며,
# `open()`함수는 컨텍스트 관리자 프로토콜을 구현하고 있으므로, 블록이 끝나거나 예외가 발생한 경우에 모두 파일이 자동으로 닫힙니다.
# 
# 컨텍스트 관리자는 `__enter__` 와 `__exit__` 두 개의 매직 메서드로 구성된다. 
# - `__enter__`: `with` 문에 진입할 때 실행되며, 반환값이 `as` 키워드의 변수에 할당됨.
# - `__exit__` : `with` 블록이 끝날 때 실행되며, 예외가 발생한 경우에도 호출됨.

# # <mark>실습</mark> `Matrix` 클래스 구현
# 2차원 데이터를 다룰 수 있는 `Matrix` 클래스를 구현해 봅니다.
# 
# 1. `__init__(self, data)`
#    - `Matrix` 클래스는 중첩리스트(`list[list[int]]`, nested list)를 기반으로 하는 행렬 클래스입니다.
#    - 생성자에서는 전달받은 중첩리스트를 인스턴스 속성(Attribute)으로 저장합니다.
#    - 깊은 복사(deep copy) 수행하여 원본 데이터의 변경이 `Matrix` 인스턴스에 영향을 미치지 않도록 합니다
# 
# 2. `__getitem__(self, index)`
#    - 행렬의 특정 행 또는 개별 요소에 접근하는 기능을 제공합니다.
#    - **행 단위 접근**: `matrix[row]` → 해당 행(row)의 모든 요소를 리스트로 반환합니다.  
#    - **요소 단위 접근**: `matrix[row, col]` → 해당 요소(row, col)를 반환합니다. 
#      - 첫번째 인덱스가 행(row), 두 번째 인덱스가 열(column)입니다. 즉, `matrix[row][col]`형태로 인덱싱합니다.
#    - 예시:
#         ```python
#         matrix = Matrix([
#             [1, 2, 3],      # Row 0
#             [4, 5, 6],      # Row 1
#             [7, 8, 9]       # Row 2
#         ])
#         print(matrix[1])       # 출력: [4, 5, 6]  (행 반환)
#         print(matrix[1, 2])    # 출력: 6  (요소 반환)
#         ``` 
# 
# 3. `__call__(self, scalar)`
#    - 행렬의 모든 요소에 주어진 `scalar` 값을 곱하여 기존 행렬 데이터를 갱신합니다.
#    - list comprehension을 사용하여 코드를 1줄로 작성합니다
#    - 예시:
#         ``` python
#         matrix = Matrix([[1, 2], [3, 4]])
#         matrix(2)
#         print(matrix.data)
#         # 출력: [[2, 4], [6, 8]]
#         ```
# 
# 4. `get_elements_greater_than(self, threshold)`
#    - `threshold`보다 큰 요소들만 골라 1차원 리스트로 반환합니다.
#    - list comprehension을 사용하여 코드를 1줄로 작성합니다
#    - 예시:
#         ``` python
#         matrix = Matrix([[1, 2, 5], [6, 3, 8]])
#         print(matrix.get_elements_greater_than(4))
#         # 출력: [5, 6, 8]
#         ```
# 5. `get_first_n_rows(self, n)`
#    - 처음 `n`개의 행을 골라 중첩리스트(`list[list[int]]`)로 반환합니다.
#    - 예시: 
#         ```python
#         matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         print(matrix.get_first_n_rows(2))
#         # 출력: 
#         # [
#         #   [1, 2, 3],
#         #   [4, 5, 6]
#         # ]
#         ```
# 
# 6. `get_first_n_columns(self, n)`
#    - 처음 `n`개의 열을 골라 중첩리스트(`list[list[int]]`)로 반환합니다.
#    - 예시: 
#         ```python
#         matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#         print(matrix.get_first_n_columns(2))
#         # 출력: 
#         # [
#         #   [1, 2],
#         #   [4, 5],
#         #   [7, 8]
#         # ]
#         ``` 

# In[ ]:


import copy

class Matrix:
    """2차원 행렬을 다루는 클래스"""

    def __init__(self, data):
        """
        전달받은 데이터를 깊은 복사하여 저장한다.

        Args:
            data (list[list[int]]): 행렬 데이터를 담은 중첩 리스트
        """
        self.data = copy.deepcopy(data)      # TODO

    def __getitem__(self, index):
        """
        행렬의 특정 행 또는 개별 요소에 접근하는 기능을 제공합니다.
        
        Args:  
            index (int) → 해당 행(row)의 모든 요소를 리스트로 반환합니다.
            index`(tuple[int, int]) → 해당 요소(row, col)를 반환합니다.

        Returns:
            list[int] OR int
        """
        if isinstance(index, tuple):        # Element-wise access
            row, col = index
            value = self.data[row][col]     # TODO
        elif isinstance(index, int):        # Row-wise access
            value = self.data[index]         # TODO

        return value
    

    def __call__(self, scalar):
        """
        행렬의 모든 요소에 주어진 스칼라 값을 곱하여 기존 행렬 데이터를 갱신합니다.
        
        Args:
            scalar (int): scalar value to multiply by
        """
        self.data = [[x * scalar for x in row] for row in self.data]  # TODO. List comprehension을 이용하여 1줄로 작성하세요

    def get_elements_greater_than(self, threshold):
        """threshold보다 큰 요소들만 골라 1차원 리스트로 반환합니다."""
        filtered_values_list = [x for row in self.data for x in row if x > threshold]  # TODO
        return filtered_values_list

    def get_first_n_rows(self, n):
        """처음 n개의 행을 골라 중첩리스트(list[list[int]])로 반환합니다."""
        sliced_data = self.data[:n]  # TODO
        return sliced_data

    def get_first_n_columns(self, n):
        """처음 n개의 열을 골라 중첩리스트(list[list[int]])로 반환합니다."""
        sliced_data =  [row[:n] for row in self.data]  # TODO. List comprehension을 이용하여 1줄로 작성하세요
        return sliced_data


# 아래 셀을 통하여 구현결과를 테스트 해보세요

# # <mark>실습</mark> `Timer` 클래스 구현
# 
# `Timer` 클래스는 코드 실행 시간을 자동으로 측정해주는 컨텍스트 매니저(context manager) 입니다.
# 이 클래스를 사용하면 `with` 문을 활용하여 특정 코드 블록의 실행 시간을 자동으로 기록하고 출력할 수 있습니다.
# 
# ## 실행 예시
# 
# ``` python
# with Timer() as t:
#     time.sleep(1) # 실행할 코드 블럭
# 
# # 예상 출력: Execution Time: 1.000xxx seconds
# ```
# 
# 힌트: `time.time()` 함수를 이용하세요 ([docs](https://docs.python.org/ko/3.13/library/time.html#time.time))

# In[ ]:


import time

class Timer:
    """코드 실행 시간을 측정하는 컨텍스트 매니저"""
    def __enter__(self):
        """ 실행 시작 시간을 저장합니다. """
        self.start_time = time.time()   # TODO
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 실행 종료 시간을 저장하고 경과 시간을 초(Seconds) 단위로 계산하여 출력 합니다."""
        self.end_time = time.time()    # TODO
        self.elapsed_time = self.end_time - self.start_time    # TODO
        print(f"Execution Time: {self.elapsed_time:.6f} seconds")


# In[ ]:




