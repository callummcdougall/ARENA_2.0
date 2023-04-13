import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_two_lines(x1=None, y1=None, x2=None, y2=None, name1="", name2="", title="", xaxis="", yaxis=""):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if x1 is None: x1 = list(range(len(y1)))
    if x2 is None: x2 = list(range(len(y2)))
    fig.add_trace(go.Scatter(x=x1, y=y1, name=name1), secondary_y=False)
    fig.add_trace(go.Scatter(x=x2, y=y2, name=name2), secondary_y=True)
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()