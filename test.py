import datetime

date_time_obj = datetime.datetime.strptime('2022-02-11', '%Y-%m-%d')


date_today = datetime.datetime.strptime('2022-02-11', '%Y-%m-%d')
date_five_years_ago = date_today - datetime.timedelta(days=round(365.25 * 5))


print(date_five_years_ago)