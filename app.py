import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


app = dash.Dash(__name__)


app.layout = html.Div(
html.Div([
    html.Div([
        html.H1('Credit Scoring Ensemble Dash-Board', style={
            'textAlign': 'center',
            'color': '#726D6C',
        }),
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[
            {'label': 'RandomForest', 'value': 'RandF'},
            {'label': 'Adaboost + SVM', 'value': 'adaB_SVM'},
            {'label': 'Adaboost + DT', 'value': 'adaB_DT'}
        ]
                
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label' : 'Majority_Voting', 'value' : 'MV'}],
                
            ),
            
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]
        ),



    
])
       
            )


def update_graph():
    

    return {
        
    }


if __name__ == '__main__':
    app.run_server(debug=True)
