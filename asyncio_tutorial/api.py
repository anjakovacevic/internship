# documentation for asyncio: https://docs.python.org/3/library/asyncio.html

import asyncio

async def fetch_data()->str:
    print('Fetching data..')
    await asyncio.sleep(2.5)  # Cekaj pre nego sto izvrsis sledecu liniju
    print('Data Fetched!')
    return 'API Data'