from sqlalchemy import create_engine, MetaData, Table, select, inspect
from sqlalchemy.orm import sessionmaker

import math
import logging
import traceback
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from dateutil.relativedelta import *
import numpy as np
from scipy.signal import argrelextrema

## Login details
username = 'Lab'
password = 'Notepad12-Table18'
host = '10.0.132.75'
port = '3306'

class Handler:
    logger = logging.getLogger('CT_Database_Server')

    def __init__(self, databaseName='mydata'):
        # Create the engine for interacting with the database
        self.engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{databaseName}", pool_pre_ping=True, pool_recycle=60)
        self.Session = sessionmaker(self.engine)
        
        ## Load the tables from the database
        self.metadata_obj = MetaData()

        # Get the tag name table
        self.tagTable = Table("sqlth_te", self.metadata_obj, autoload_with=self.engine)

        ## Read all of the data tables
        tables = inspect(self.engine).get_table_names()
        dataTables = [x for x in tables if 'sqlt_data' in x]
        self.dataTables = {}
        for tableName in dataTables:
            index = dt.strptime(tableName, "sqlt_data_%d_%Y_%m").strftime("%Y-%m")
            self.dataTables[index] = Table(tableName, self.metadata_obj, autoload_with=self.engine)

        ## Get the most recent date table
        self.dataTable = self.dataTables[max(self.dataTables.keys())]
        #self.dataTable = Table("sqlt_data_1_2024_11", self.metadata_obj, autoload_with=self.engine)


    def getTagIDDictionary(self):
        ''' Get a dictionary with each tag ID and name for lookup '''
        with self.Session() as session:
            # Query all of the tag IDs and paths
            query = session.query(self.tagTable.c.id, self.tagTable.c.tagpath).all()
        
        # Format into a dictionary and return
        result = {id:name for (id, name) in query}
        return result
    

    def queryTableBetween(self, table, start:dt, end:dt, tagids=None):
        ''' Query data from a single table in the database from start to end date '''

        # Convert the start and end times to millisecond epoch time
        startTimestamp = start.timestamp() * 1000
        endTimestamp = end.timestamp() * 1000

        # Build query
        query = select(table).filter(table.c.t_stamp >= startTimestamp, table.c.t_stamp <= endTimestamp)
        if(tagids != None and type(tagids) == list):
            query = query.filter(table.c.tagid.in_(tagids))

        # Get data into pandas dataframe
        with self.engine.connect() as conn:
            data = pd.read_sql_query(query, conn)

        # Get the name for each tagID mapped on
        data['tagid'] = data['tagid'].map(self.getTagIDDictionary())

        # Reindex to datetime column
        newdf = data.set_index('t_stamp')

        # Empty new dataframe to fill
        df = pd.DataFrame()

        # Merge all the tags into one dataframe with datetime as index, each a unique column
        for tag in newdf['tagid'].unique():
            if(not newdf[newdf['tagid'] == tag]['floatvalue'].isnull().all()):
                tagdata = newdf[newdf['tagid'] == tag][['floatvalue']].rename(columns={'floatvalue': tag})
            elif(not newdf[newdf['tagid'] == tag]['intvalue'].isnull().all()):
                tagdata = newdf[newdf['tagid'] == tag][['intvalue']].rename(columns={'intvalue': tag})
            elif(newdf[newdf['tagid'] == tag]['stringvalue'].isnull().all()):
                tagdata = newdf[newdf['tagid'] == tag][['stringvalue']].rename(columns={'stringvalue': tag})
            else:
                continue
            
            df = df.join(tagdata, how='outer')

        if(df.empty):
            return None

        # Sort by datetime, calculate hours since start, and reset index to incrementing integers
        df.reset_index(inplace=True)
        df = df.sort_values(by='t_stamp')

        return df


    def getDataframeBetween(self, start:dt, end:dt, tagids=None):
        ''' Given a start and end datetime, return a dataframe of the tagids data.
        
         Includes processing for merging multiple data month tables together '''

        if(start < dt.strptime(min(self.dataTables.keys()), '%Y-%m')):
            return "Start given is before logging started"
        if(end > (dt.strptime(max(self.dataTables.keys()), '%Y-%m') + relativedelta(months=1) - timedelta(days=1) + timedelta(hours=23, minutes=59, seconds=59))):
            return "End given is after logging started"

        ## Find which table this data is coming from
        startTable = self.dataTables[start.strftime("%Y-%m")]
        endTable = self.dataTables[end.strftime("%Y-%m")]
        if(startTable == endTable):
            queryTable = startTable
            df = self.queryTableBetween(queryTable, start, end, tagids)
        
        else:
            # Empty df to fill
            df = pd.DataFrame()

            ## Query the start table
            lastDayOfMonth = (dt(start.year, (start + relativedelta(months=1)).month, 1) - timedelta(days=1)).day
            newend = dt(start.year, start.month, lastDayOfMonth, 23, 59, 59)
            newdf = self.queryTableBetween(startTable,  start, newend, tagids)
            df = pd.concat([df, newdf])

            ## Query the end table
            newstart = dt(end.year, end.month, 1, 0, 0, 0)
            newdf = self.queryTableBetween(endTable, newstart, end, tagids)
            df = pd.concat([df, newdf])

            ## Query any middle tables
            monthsBetween = (end.year - start.year) * 12 + end.month - start.month
            if(monthsBetween > 1):
                for i in range(1, monthsBetween):
                    ## Start of this month
                    tmp = start + relativedelta(months=i)
                    newstart = dt(tmp.year, tmp.month, 1, 00, 00, 00)

                    ## End of this month
                    firstDayNextMonth = dt(newstart.year, (newstart + relativedelta(months=1)).month, 1)
                    lastDayOfMonth = (firstDayNextMonth - timedelta(days=1)).day
                    newend = dt(newstart.year, newstart.month, lastDayOfMonth, 23, 59, 59)
                    
                    ## Get data
                    table = self.dataTables[newstart.strftime('%Y-%m')]
                    newdf = self.queryTableBetween(table, newstart, newend, tagids)
                    df = pd.concat([df, newdf])
        
        if(type(df) == type(None)):
            return None

        ## Get datetime and return dataframe
        df['DateTime'] = pd.to_datetime(df['t_stamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        df['DateTime'] = df['DateTime'].dt.tz_localize(None)

        ## Get data from multiple tables
        return df


    def getHourlyRateOfChange(self, tags):
        ''' Get the last hour rate of change data for tags '''
        try:
            # Get the tag data
            start = ((dt.now() - timedelta(hours=1)).timestamp()) * 1000
            end = ((dt.now()).timestamp()) * 1000

            # Query data
            with self.Session() as session:
                query = session.query(self.dataTable.c.tagid, self.dataTable.c.t_stamp, self.dataTable.c.floatvalue).filter(self.dataTable.c.tagid.in_(tags)).filter(self.dataTable.c.t_stamp < end).filter(self.dataTable.c.t_stamp > start)
            if(query.all() == []):
                return 'No Data Found'

            ## Convert to dataframe
            df = pd.DataFrame(query.all())
            df['DateTime'] = pd.to_datetime(df['t_stamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            df['DateTime'] = df['DateTime'].dt.tz_localize(None)

            ## Split dataframe by tag and find peaks and clean and calculate rate of change
            dfs = {}
            order = 75
            for tagid in df['tagid'].unique():
                # Subset
                dfs[tagid] = df[df['tagid'] == tagid]

                # Find local maxima
                dfs[tagid]['max'] = dfs[tagid].iloc[argrelextrema(dfs[tagid].floatvalue.values, np.greater_equal, order=order)[0]]['floatvalue']

                ## Check that this is an actual peak and not just a trailing increase
                for index, row in dfs[tagid].dropna(subset=['max']).iterrows():
                    if(index > (dfs[tagid].index.max() - 10)):
                        dfs[tagid].loc[index, 'max'] = np.nan
                
                # Clean data and exclude any too close together
                dfs[tagid] = dfs[tagid][['DateTime', 'max']].dropna()
                dfs[tagid] = dfs[tagid][dfs[tagid]['DateTime'].diff().dt.total_seconds() > 100]

                # Calculate rate of change
                dfs[tagid]['dt'] = pd.to_numeric(dfs[tagid]['DateTime'].diff().dt.total_seconds())
                dfs[tagid]['dmax'] = pd.to_numeric(dfs[tagid]['max'].diff())
                dfs[tagid]['roc'] = (dfs[tagid]['dmax'] / dfs[tagid]['dt']) * 60

            return dfs

        except:
            return traceback.format_exc()
        

    def getPeakToPeakLastHour(self, tag):
        ''' Get the last hour peak to peak for the specified tag '''
        # Get the tag data
        start = ((dt.now() - timedelta(hours=1)).timestamp()) * 1000
        end = ((dt.now()).timestamp()) * 1000

        # Query data
        with self.Session() as session:
            query = session.query(self.dataTable.c.tagid, self.dataTable.c.t_stamp, self.dataTable.c.floatvalue).filter(self.dataTable.c.tagid == tag).filter(self.dataTable.c.t_stamp < end).filter(self.dataTable.c.t_stamp > start)

        if(query.all() != []):
            ## Convert to dataframe
            df = pd.DataFrame(query.all())
            df['DateTime'] = pd.to_datetime(df['t_stamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            df['DateTime'] = df['DateTime'].dt.tz_localize(None)

            ## Split dataframe by tag and find peaks and clean and calculate rate of change
            order = 75

            # Find local maxima
            df['max'] = df.iloc[argrelextrema(df.floatvalue.values, np.greater_equal, order=order)[0]]['floatvalue']
            df['min'] = df.iloc[argrelextrema(df.floatvalue.values, np.less_equal, order=order)[0]]['floatvalue']

            ## Check if this max/min is just on a ramp up or down
            for index, row in df.dropna(subset=['max']).iterrows():
                if(index > (df.index.max() - 10)):
                    df.loc[index, 'max'] = np.nan
            for index, row in df.dropna(subset=['min']).iterrows():
                if(index > (df.index.max() - 10)):
                    df.loc[index, 'min'] = np.nan

            mins = df[['DateTime', 'min']].dropna()
            maxs = df[['DateTime', 'max']].dropna()

            peak_to_peaks = []
            for index, row in mins.iterrows():
                nearestMaxRow = maxs.iloc[(pd.Series(maxs.index) - index).abs().argsort()[:2]]
                max = nearestMaxRow['max']
                min = row['min']
                diff = (max - min).mean()
                time = row['DateTime']

                peak_to_peaks.append([time, diff])

            return peak_to_peaks
        
        else:
            return 'Failed to get data'