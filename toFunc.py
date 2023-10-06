def ComputeStepSleep(x, t, df, mV):
    '''
    Compute number of steps during a sleep window and the sleep duration
    x = dfNoContra
    t = dfUTC
    df = (corrected) raw dataset 
    mV = missVal
    '''        
    dfDummy = pd.DataFrame({'sid': [], 'night': [], 'step_number': [], 'sleep_duration': []})

    for sid in x['sid'].values:
        df_diff = {'sid': [], 'night': [], 'step_number': [], 'sleep_duration': []}
        max_night = x[x.sid == sid]['max_night'].values[0]
        mtNight = x[x.sid == sid].empt_night.values[0] # This returns the list as stored in the dataframe
        # map(int, re.findall(r'\d+', mtNight))
        if (mV != []) & (sid in mV):
            mtNight = mV[sid] + mtNight

        for night in range(1, max_night+1):
            if night not in mtNight:
                valOn = t[(t['series_id'] == sid) & (t['night'] == night) & (t['event'] == 'onset')]['step'].values[0]
                valUp = t[(t['series_id'] == sid) & (t['night'] == night) & (t['event'] == 'wakeup')]['step'].values[0]
                diff = valUp - valOn

                df_diff['sid'].append(sid)
                df_diff['night'].append(night)
                df_diff['step_number'].append(diff)

                valOn = pd.to_datetime(df[(df.series_id == sid) & (df.night == night) & (df.event == 'onset')]['timestamp'], utc=True)
                valUp = pd.to_datetime(df[(df.series_id == sid) & (df.night == night) & (df.event == 'wakeup')]['timestamp'], utc=True)
                diff = pd.Series(valUp.values - valOn.values, name='duration')
        
                df_diff['sleep_duration'].append(diff)
    
        temp = pd.DataFrame(df_diff)
        dfDummy = pd.concat([dfDummy, temp], ignore_index=True)
        del(temp)

    
    dfDummy = dfDummy.astype({'night': 'int8'})
    dfDummy = dfDummy.astype({'step_number': 'int16'})
    # df_diff.to_csv('differences.csv', index=False)