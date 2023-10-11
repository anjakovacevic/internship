import api
import asyncio

async def send_data(to:str):
    print(f'Sending data to {to}...')
    await asyncio.sleep(2)
    print(f'Data sent to {to}!')

async def main():
    # we don't wanna proceed with the program until this data is fetched!
    data = await api.fetch_data()
    print('Data: ', data)
    
    # another option is to create a list of functions and send it to the gather function
    # this allows for the data to be sent to both mario and luigi at the same time
    await asyncio.gather(send_data('Mario'), send_data('Luigi'))

if __name__ == '__main__':
    asyncio.run(main())