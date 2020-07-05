from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import config, neuralnetwork_tech, papertrade


default_args = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2020, 6, 17),
    'email': ['khalil.mejouate@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'schedule_interval': '*/5 * * * *',
}

stock = 'DIA'
time_window = 'daily'

dag = DAG('alpaca-trading_dag', catchup=False, default_args=default_args, dagrun_timeout=timedelta(seconds=5))

def load(stock, time_window):
    config.save_dataset(stock, time_window)
    neuralnetwork_tech.Build_Model(stock, time_window)
def trade(stock, time_window):
    papertrade.buy_sell(stock, time_window)

load_task = PythonOperator(task_id='loading',
                          python_callable=load,
                          dag=dag)
trade_task = PythonOperator(task_id='trading',
                            python_callable=trade,
                            dag=dag)
trade_task.set_upstream(load_task)

# trade(stock, time_window)