# SHARK ATTACK PROJECT

## Objetivo do projeto

O objetivo deste projeto é tentar entender um pouco de como as atividades praticadas no mar podem influenciar no comportamento dos tubarões com relação às motivações e intensidades do ataque desses animais.

## Banco de dados

O estudo foi realizado em cima do banco de dados presente no link: https://www.sharkattackfile.net/whystudy.htm que contém milhares de informações detalhadas a respeito de casos de ataques de tubarões. 
O projeto elaborado aqui tenta, ao máximo, preservar as mesmas definições utilizadas no link para que haja uma sinergia de informações.

## O passo a passo

1. Padronização de nome de colunas
2. Limpeza geral dos dados
3. Pergunta 1: Existem atividades que mais provocam os tubarões?
4. Pergunta 2: O índice de fatalidade tem relação com o tipo de ocorrência dos casos?
5. Considerações finais

## Metodologias utilizadas

Para o projeto foram utilizadas:
- bibliotecas pandas e regex
- for loops
- métodos de acesso, filtros e máscaras e agrupamentos em dataframe
- meta caracteres, quantificadores em regex
- uso de funções junto ao apply

## O processo

### Padronização de nome de colunas
Nesta primeira etapa do processo foi realizado um código para que o nome de todas as colunas fossem alteradas para manterem um certo padrão.

O foco foi, principalmente, em transformar as letras maiusculas em minusculas e substituir os espaços para underlines, visto que facilita o manuseio da nossa base. 
Por meio do método de regex dentro de uma for loop, foi possível identificar estes caracteres para cada nome de coluna e realizar as devidas alterações.


### Limpeza geral dos dados
Foi percebido um grande volume de dados vazios na dataframe.

De início, observamos duas colunas "unnamed_22" e "unnamed_23" que além de não serem nomeadas, também não tinham conteúdo relevante para os dados. Logo, elas foram dropadas da nossa base.

Depois disso, foi visto que muitas linhas também estavam vazias e/ou preenchidas com valores que não estavam dentro dos padrões de outras respostas. Então foi determinado que as linhas precisaria de **pelo menos** 3 informações completas para seja válido a sua permanência na base. Está correção foi possível com o uso de um dropna() e thresh de 3.


### Pergunta 1: Existem atividades que mais provocam os tubarões?

O primeiro tema questionado foi se existiam algumas atividades que mais ameaçavam e provocavam os tubarões.

Para este estudo foram-se utilizados as colunas 'type', que relata o caráter da ocorrência (Watercraft, Unprovoked, Questionable, Provoked, Sea Disaster')
e a coluna 'activity', que indica o que as vítimas estavam fazendo na hora em que foram atacadas.


#### Limpeza da coluna 'type'
Os dados inputados da base não estavam coerentes com os valores no log de incidência do site Shark Attack File e, portanto, realizou-se uma manipulação de dados para tornar tudo padronizado com a documentação.

Dados relacionados à embarcação ('Watercraft') estavam todos indicados como 'Boating' e variações escritas erradas. Para isso, um apply com função regex.sub foi utilizada para transformar todos os valores relacionados a Boat para 'Watercraft'.

Dados classificados como 'Invalid', foram transformados para 'Questionable' utilizando-se do mesmo método.

Colunas nulas foram considerados como não definidos e, por não haver volumes representativos, não foram alterados.




#### Limpeza da coluna 'activity'
A coluna 'activity' possuía dados muito distintos entre eles, o que tornou mais difícil a padronização dos dados.
Olhando os dados foi percebido alguns termos que eram mais presentes nos relatos de atividades e elas foram utilizadas, então, para categorizar as atividades. Foram elas:

'Surfing', 'Fishing', 'Swimming', 'Diving', 'Wading', 'Bathing', 'Standing', 'Boating', 'Snorkeling', 'Fell in water', 'Floating', 'Paddling' e 'Messing with sharks'

Para encontrar as linhas com estes termos foi utilizada uma função com apply na dataframe que pegassem as células que continham o padrão regex .*[Pp]alvra.*
Encontradas as células, mais uma função aplicada para realizar a substituição pelas palavras da lista acima.

#### Relação 'type' e 'activity'

Tendo as atividades categorizadas, podemos agora cruzar as duas colunas para identificar quais delas acabam sendo mais provocativas aos tubarões.

As 10 atividades com maior ocorrência de ataques de tubarão são:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Surfing</th>
      <td>21.723262</td>
    </tr>
    <tr>
      <th>Fishing</th>
      <td>18.121231</td>
    </tr>
    <tr>
      <th>Swimming</th>
      <td>17.597588</td>
    </tr>
    <tr>
      <th>Non defined</th>
      <td>8.632180</td>
    </tr>
    <tr>
      <th>Diving</th>
      <td>8.108537</td>
    </tr>
    <tr>
      <th>Wading</th>
      <td>3.284672</td>
    </tr>
    <tr>
      <th>Bathing</th>
      <td>3.030784</td>
    </tr>
    <tr>
      <th>Standing</th>
      <td>2.475405</td>
    </tr>
    <tr>
      <th>Boating</th>
      <td>2.300857</td>
    </tr>
    <tr>
      <th>Messing with sharks</th>
      <td>1.904157</td>
    </tr>
  </tbody>
</table>

Quando filtramos as ocorrências pelos casos 'Provoked' temos que:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fishing</th>
      <td>54.878049</td>
    </tr>
    <tr>
      <th>Messing with sharks</th>
      <td>14.111498</td>
    </tr>
    <tr>
      <th>Diving</th>
      <td>7.839721</td>
    </tr>
    <tr>
      <th>Non defined</th>
      <td>6.097561</td>
    </tr>
    <tr>
      <th>Surfing</th>
      <td>3.658537</td>
    </tr>
    <tr>
      <th>Wading</th>
      <td>2.090592</td>
    </tr>
    <tr>
      <th>Boating</th>
      <td>1.916376</td>
    </tr>
    <tr>
      <th>Swimming</th>
      <td>1.916376</td>
    </tr>
    <tr>
      <th>Standing</th>
      <td>1.742160</td>
    </tr>
    <tr>
      <th>Snorkeling</th>
      <td>0.871080</td>
    </tr>
  </tbody>
</table>

Comparando as duas informações é possível inferir que mais da metade dos ataques provocados ocorrem na pesca. A proporção quando comparada ao geral é 36% maior.

Algumas hipóteses levantadas são de que a pesca se torna perigosa, pois as vítimas estavam caçando os próprios tubarões, os tubarões foram atraídos pelos peixes já pescados pelos nadadores ou a autodefesa da pessoa quando se vê em meio dos tubarões. Para entender com mais profundidade, seria necessário um deep dive nas respostas.

Outro ponto interessante é que 'Messing with sharks' não apresenta tanta relevância no geral (~2%), mas quando filtramos os casos provocativos, vemos que se trata do segundo maior motivo de ataque(~14%).


### Pergunta 2: O índice de fatalidade tem relação com o tipo de ocorrência dos casos?

O segundo tema foi tentar entender se o tipo de ocorrência acabava feridas que fossem mais letais às vidas das vítimas. Para isso, foi feita a limpeza da coluna fatal_yn para ser cruzada ao type.

#### Limpeza type e fatal_yn

A coluna 'fatal_yn' não possuía valores muito distintos, foi necessário apenas alguns pequenos reparos de erros de digitação para a padronização. 

Alguns valores como '2017', 'M' ou 'y' que apareceram apenas uma vez foram alterados pelo recurso do loc.

Outros erros como ' N' que estiveram mais presentes foram corrigidos com o uso de uma substituição de regex e uma função apply na dataframe.

No meio deste processo, foi percebido que algumas células de 'fatal_yn' com dados nulos tinham a indicação de fatalidade na coluna de 'injury'. Desse modo, utilizando-se do regex novamente, foi criado uma nova coluna 'fatal_injuries' para incluir as células que contivessem FATAL no texto.

As células nulas de 'fatal_yn' com fatal em 'injury' foram, então, substituídas por Y indicado pelas células FATAL na coluna 'fatal_injuries'. As células nulas restantes foram substituídas por N, pois não se tratavam de feridas fatais.

A coluna 'fatal_yn' também tinha valores 'UNKNOWN' que foram considerados como dados inválidos.

#### Relação 'type' e 'fatal_yn'
Com a coluna 'fatal_yn' limpa, ela foi transformada por uma função onde os valores 'Y' se tornavam 1, 'N' se tornava 0 e 'UNKNOWN' se tornava None e armazenada em uma nova coluna 'fatal_binary'.

Esse processo foi realizado para que fosse possível realizar funções numéricas ao cruzamento de dados.

Tendo transformado os dados, foi realizado um agrupamento das colunas 'type' e 'fatal_binary', tendo a primeira como referência. Na 'fatal_binary' foi realizado a soma para cada 'type' para identificar as fatalidades em cada uma delas.

Por fim, chegamos a este resultado:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fatal_ocurrencies</th>
      <th>total_ocurrencies</th>
      <th>fatal_proportion</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Provoked</th>
      <td>19.0</td>
      <td>574.0</td>
      <td>3.310105</td>
    </tr>
    <tr>
      <th>Questionable</th>
      <td>59.0</td>
      <td>549.0</td>
      <td>10.746812</td>
    </tr>
    <tr>
      <th>Sea Disaster</th>
      <td>168.0</td>
      <td>239.0</td>
      <td>70.292887</td>
    </tr>
    <tr>
      <th>Unprovoked</th>
      <td>1183.0</td>
      <td>4595.0</td>
      <td>25.745375</td>
    </tr>
    <tr>
      <th>Watercraft</th>
      <td>11.0</td>
      <td>341.0</td>
      <td>3.225806</td>
    </tr>
    <tr>
      <th>nan</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>

Não houve uma relação direta entre a provocação dos tubarões e o índice de fatalidade dos ataques provocados.

'Watercraft' apresenta valores baixos (~3.31%), pois muitos dos ataques de tubarões neste tipo de ocorrência não foram diretamente à vítima, mas ao equipamento sendo utilizado por ela (barco, remo, etc).

Observamos também que Sea Disaster aparece com bastante relevância (~70.3%) e uma das hipóteses é de que se tratam de pessoas que caíram de suas embarcações ou que o navio que estavam naufragou não tendo escapatória.

## Considerações finais
Chegamos a conclusão de que existem, sim, atividades que provocam mais os tubarões, estão entre elas a pesca, tanto de tubarões quanto de outro animais e o ato de brincar, incomodar os tubarões. 
Mas também vimos de que isso não quer dizer que o ataque será intensa a ponto de chegar a fatalidade, pois os índices são baixos quando os tubarões são provocados. 

Para um melhor entendimento qualitativo do problema, será necessário um tempo maior dedicado ao estudo dos dados inputados que pode ser realizado para o próximo projeto.



